# The three ablation arms — what the driver can now express, and what each arm costs

D-034 schedules three arms before the results freeze. Until this branch the
driver could express none of them: the protocol was fixed at horizon 1,
window 500, step 1, `refit_every` 21, daily re-conditioning and each asset's
own primary scoring target, and `--help` offered no way to move any of it.

| arm | what moves | flag | assets |
|---|---|---|---|
| 1 | fit window 500 → 1000 | `--window 1000` | all 11 |
| 2 | re-conditioning between refits, daily → frozen | `--recondition none` | all 11 |
| 3 | scoring target → Parkinson, Garman–Klass, squared close-to-close | `--target ...` | the 9 equity assets |

Arm 1 is D-019's robustness arm against the 500 primary. Arm 2 is D-011's
efficiency reading and the control for D-015's "re-estimate every N,
re-condition daily". Arm 3 is D-017's *labelled* robustness proxy, on equities
only: crypto's primary is 5-minute realized variance (D-004) and a 24/7 market
has no overnight session for a close-to-close estimator to measure.

**This document covers the code change and the measurements. It does not run
the arms.** The launch commands are §9.

---

## 1. What the flags are, and what they are not

Three flags on `volbench.benchmarks.grid_primary`, named for the objects they
set rather than for the arms they serve:

- **`--window`** (default 500, D-019) — a field of `ProtocolArm`, handed to
  `RollingOriginSplitter`, and therefore in every cell's config hash through
  `splitter.window`.
- **`--recondition`** (default `daily`, D-015) — a field of `ProtocolArm`,
  handed to `run_backtest`, recorded under `protocol.recondition` whenever
  `refit_every > 1`.
- **`--target`** (default: each asset's own primary) — the *cell's* scoring
  proxy, not a protocol setting, because D-017 makes the target a property of
  the evaluation cell. It reaches the hash through `proxy_name` and the
  proxy's content digest.

**They feed the existing plumbing; there is no second path.** Every one of
these settings already reached the config hash before this branch — the arms
were unreachable from the command line, not unrepresentable in the store. The
flags therefore build the same `ProtocolArm` the study has always built
(`grid_primary.protocol_arm`) and select a column of the same `targets` frame
the panel has always produced (`grid_primary.asset_data`). Nothing was added
to `results.build_config`, to `runner`, or to `evaluate`. This is the failure
D-032 spent a day on: a setting written down somewhere the hash does not see
produces fragments whose hash does not describe them.

**`--target` never changes what a model is fed.** D-017 is explicit that the
fit input is a modelling contract: a variance-fed model is fitted on the
asset's own primary series whatever the cell is scored against, because the
input defines what the model forecasts and the arm's whole purpose is to
re-score *the same forecasts* against a second proxy. §6 measures that this
is what actually happens, at the level of the stored rows.

**The arm's label says what moved.** `protocol_arm()` with no argument is the
headline arm, label and all; each flag that moves appends to the label
(`w1000`, `recondition-none`, `target-parkinson`). The label is the only
per-cell field in the committed manifest free to carry the scoring target —
`CellOutcome`'s keys are fixed and enumerated by
`tests/test_manifest_provenance.py` — so a reader of
`docs/P3_GRID_manifest_<tag>.json` can see what its cells were scored against
without opening a store they do not have. The label is not hashed; the
settings it names all are.

**Crypto is refused a range target, before the panel is read.** `--target`
names one of the four range/close-to-close estimators, and D-004 keeps BTC-USD
and ETH-USD on realized variance. Asking for a whole-panel target run is a
`parser.error` naming the two assets, D-004, and `--assets` — raised from the
declared panel membership (`CRYPTO_PANEL`) so it costs a second rather than
the ~60 s `build_panel` takes.

## 2. The defaults reproduce the primary grid

The property that makes any of this safe: a run with none of the new flags is
the run that produced `docs/P3_GRID_manifest.json`.

```
uv run --no-sync python -m volbench.benchmarks.grid_primary \
    --assets SPY --models garch11 --tag defaults_reproduce --out-dir <tmp>
```

into an **empty** store, so the cell was recomputed from the archives rather
than read back:

| | |
|---|---|
| config hash | `4968957865bf5159dd37e5b3799313fd7707c25327b5712460a1e1347e5f3a80` |
| committed manifest's SPY/`garch11` hash | the same |
| arm label | `headline` |
| rows / fits / fallback / non-converged | 4,904 / 234 / 1 / 1 — the manifest's figures |
| `sha256` of the recomputed `.parquet` | `2b710dfe…`, **identical to the fragment in the study store** |
| `sha256` of the recomputed `.json` sidecar | `6e93f363…`, identical |

Byte-identical, not merely hash-identical. The flags disturbed nothing.

## 3. The hash tests, both halves

`tests/test_grid_primary_arms.py`, 28 tests. Each flag is checked twice,
because a flag that *always* moves the hash is as broken as one that never
does: the first fragments a store into cells nothing can find again, the
second serves one arm's fragments for another's.

| flag | moves the hash | leaves it alone |
|---|---|---|
| `--window` | `--window 1000` shares no hash with the default run, on every cell | `--window 500` reproduces the default run's hashes exactly |
| `--recondition` | `--recondition none` shares no hash with the default run | `--recondition daily` reproduces them exactly |
| `--target` | each of the three robustness targets shares no hash with the default run | `--target overnight_plus_range` reproduces them exactly on an equity asset |

With, beside each pair, the test that says *where* the setting landed:
`splitter.window` for the window, `protocol.recondition` for the
re-conditioning, and — for the target — that the stored config's
`data.proxy.sha256` moved while `data.fit_series_sha256` and
`data.series_sha256` did not. That last one is D-017 as a single assertion:
if it ever fails, the arm has stopped re-scoring the same forecasts and its
comparison with the primary means nothing.

The tests run on a synthetic panel, because `build_panel` reads
hand-downloaded archives under the gitignored `data/raw` and is absent in CI.
A config hash is a function of the series' *contents*, so everything the flags
touch is exercised in full regardless. The one thing only real data can check
is §2, and that is a run rather than a test: `build_panel` alone is ~60 s.

## 4. What each arm moves — counted, not assumed

**Method.** A cell's config sidecar in the store *is* the dict its hash was
taken over. Re-hashing it unmodified must reproduce the fragment's own name —
run first, over all 143 cells, as the method's own check — and re-hashing it
with exactly the field an arm moves gives that arm's hash for that cell
without running anything. Whether the arm shares the primary's fragment is
then `ResultsStore.has(derived)`.

| check | result |
|---|---|
| 143 sidecars re-hashed unmodified | **143 reproduce their own fragment name**, 0 do not |
| `--target overnight_plus_range` proxy digest vs the digest stored for each of the 9 equity assets | identical, 9/9 |
| the 15 smoke cells of §6, predicted vs. actually written | **15/15 match** |

The counts:

| arm | cells | hashes already in the primary store | hashes that move |
|---|---:|---:|---:|
| 1 · window 1000 | 143 | 0 | **143** |
| 2 · `recondition=none` | 143 | 0 | **143** |
| 3 · `target=parkinson` | 117 | 0 | **117** |
| 3 · `target=garman_klass` | 117 | 0 | **117** |
| 3 · `target=squared_return` | 117 | 0 | **117** |

637 cells across the five runs, 637 distinct hashes, no collision with each
other and none with the primary's 143.

**Arm 2 was expected to share most of its cells. It shares none, and the
reason is worth recording.** `recondition` is written into the config's
`protocol` block by `run_backtest` whenever `splitter.refit_every > 1` — for
*every* model, not only for models that can act on it. Every cell of the grid
runs at `refit_every = 21`, so every cell's hash moves. Twelve of the thirteen
configs re-condition (`FittedNaiveVol`, `FittedEWMA`, `FittedGARCH`,
`FittedHAR`, `FittedStatsForecastRV`, `FittedLightGBMRV`, `FittedTSFM` all
implement `update`); **`patchtst` is the one that cannot** — `FittedPatchTST`
has no `update`, and `run_backtest` holds its forecast between refits under
either setting. So arm 2's eleven `patchtst` cells will be *numerically
identical* to the primary's while carrying a different hash.

That is the conservative side of the trade and it is the right side: the hash
records the protocol the run was asked for, not a per-model deduction about
whether it bound. A hash that depended on which model happened to implement
`update` would move the moment an adapter gained one, and the store would
then serve a frozen-forecast fragment for a re-conditioned cell. The cost is
11 cells recomputed to prove they did not change, ≈ 27 GPU-minutes (§7); it
buys the property that no cell in the store is ever ambiguous about which arm
produced it.

## 5. Store per arm, and why

**Each arm gets its own `--out-dir`. All three share nothing with the primary
store, so there is no recomputation to avoid by sharing one.**

| arm | store | cells | shares with the primary |
|---|---|---:|---:|
| 1 | `data/grid_ablation_window1000` | 143 | 0 |
| 2 | `data/grid_ablation_recondition_none` | 143 | 0 |
| 3 | `data/grid_ablation_targets` | 351 (3 × 117) | 0 |

The argument that would have put an arm in the primary store is the one the
counts refuted: an arm sharing most of its cells would recompute a 68-minute
GPU lane for nothing. Since every arm shares zero, a separate store costs
nothing and buys two things.

**It keeps the primary's closure property intact.** The primary store holds
187 fragment pairs — the 143 the study scores and the 44 those replaced — and
`docs/P3_MANIFEST_INVENTORY.md` §4 turns "143 + 44 = 187, with no fragment
named by neither" into a checkable statement. Adding 637 arm fragments to that
store would end that sentence, and with it the ability to say which fragments
the paper's numbers come from by counting.

**It makes each arm independently deletable.** An arm that is dropped, re-run
under a changed protocol, or rebuilt after a defect is one `rm -rf` on a
directory nothing else names, rather than a set-difference over hashes.

The three targets of arm 3 share one store because they are one arm: 117 cells
each, disjoint hashes, and no reason to drop one target and keep another.

Every arm run takes **its own `--tag`**, so each writes a sibling manifest
under `docs/` and none touches `docs/P3_GRID_manifest.json`. Arm 3 is
restricted with `--assets`, which the driver already refuses under the default
tag.

## 6. The smoke runs

One asset (SPY), the cheap CPU configs (`naive`, `ewma`, `garch11`, `har`,
`lgbm`), each arm into the store it will use, each under its own tag. They are
pre-flights, not throwaways: the fragments are the arm's own, and the full run
will read them back as cache hits, exactly as the primary grid's SPY column
was read back from its pre-flight.

| smoke | arm label | rows/cell | fits (`garch11`) | cells | result |
|---|---|---:|---|---:|---|
| `smoke_window1000` | `w1000` | 4,404 | 210, 0 fallback | 5 | computed 5, failed 0 |
| `smoke_recondition_none` | `recondition-none` | 4,904 | 234, 1 fallback | 5 | computed 5, failed 0 |
| `smoke_target_parkinson` | `target-parkinson` | 4,904 | 234, 1 fallback | 5 | computed 5, failed 0 |

Every one of the 15 hashes is the hash §4 predicted for its arm, and none of
them is the primary's.

**What the stored rows say the arms actually do** — read off the fragments,
against the primary's own:

- **Arm 2 is not a no-op, and it moves exactly what it should.** On all five
  cells, `forecast_var` differs from the primary's on **4,670 of 4,904 rows =
  every non-refit origin**, and agrees exactly on the 234 refit origins. That
  is D-015's mechanism, visible: the parameters are the same, the conditional
  state is not. Median relative move on the rows that moved: 1.1% (`naive`) to
  29.5% (`har`).
- **Arm 3 re-scores and does not re-fit.** On all five cells `forecast_mean`
  and `forecast_var` are **bit-identical** to the primary's, while `proxy_var`
  and `qlike` move on all 4,904 rows. D-017, upheld at the level of the bytes.
- **Arm 1's origins are a strict subset of the primary's.** SPY: 4,904 primary
  origins, 4,404 arm origins, intersection 4,404, arm-only 0. §8.

Arm 1 got two further probes, because it is the arm whose runtime cannot be
inferred from the primary's (§7): `smoke_window1000_gpu` ran SPY's four GPU
configs and `smoke_window1000_cpu` the four remaining CPU ones, both at window
1000, into the same store. **SPY's whole 13-config column is therefore already
computed for arm 1**, and its full run starts with it cached. Every one of
those nine cells was computed with no failure and no `missing_reason` row.

## 7. Wall-clock estimates

Extrapolated from the smoke runs, and from the one thing the smoke runs alone
could not answer: **window 1000 is not the primary's runtime with 10% taken
off.** Fewer origins, but longer fits and a longer TSFM context, and the two
do not cancel. So the *whole* SPY column was measured at window 1000 — all
thirteen configs, the four GPU ones included, 9.5 min of probe — against the
window-500 timings the study already has for the same asset.

| config | lane | w500 s | w1000 s | per cell | **per origin** |
|---|---|---:|---:|---:|---:|
| `patchtst` | gpu | 179.9 | 227.6 | 1.26 | **1.41** |
| `timesfm` | gpu | 156.7 | 178.2 | 1.14 | 1.27 |
| `autoarima` | cpu | 40.5 | 62.3 | 1.54 | **1.71** |
| `chronos` | gpu | 50.5 | 51.4 | 1.02 | 1.13 |
| `moirai` | gpu | 34.4 | 34.6 | 1.00 | 1.12 |
| `lgbm` | cpu | 7.5 | 10.9 | 1.45 | 1.61 |
| `garch11_t` | cpu | 8.9 | 8.4 | 0.94 | 1.05 |
| `gjr` | cpu | 5.7 | 5.7 | 1.00 | 1.11 |
| `garch11` | cpu | 5.2 | 4.9 | 0.94 | 1.05 |
| `autoets` | cpu | 2.6 | 3.3 | 1.29 | 1.44 |
| `har` | cpu | 1.1 | 1.4 | 1.27 | 1.41 |
| `ewma` | cpu | 0.7 | 0.8 | 1.07 | 1.19 |
| `naive` | cpu | 0.6 | 0.6 | 0.91 | 1.01 |
| **column** | | **494.4** | **589.9** | **1.19** | **1.32** |

(SPY: 4,904 origins at window 500, 4,404 at 1000. The w500 column is the
current manifest's own timing where the L fix recomputed the cell, and the
13-cell pre-flight's otherwise.)

Read the last column: doubling the window costs **32% more per origin**, and
10% fewer origins gives most of it back, for **+19% per cell**. The models
that pay are the ones whose cost is in the fit or the context — `autoarima`
re-selects over twice the history, `patchtst` trains on twice the window,
`lgbm` builds twice the rows — while the three zero-shot TSFMs barely move,
their context being capped well below 1,000 anyway. The four GARCH-family and
naive cells are *cheaper*, because their per-fit cost is flat and there are
10% fewer fits (210 refits against 234).

Projected over each arm's own panel, scaling SPY's measured cell times by each
asset's origin count, CPU lane at the primary run's observed 9.1x effective
parallelism on 12 workers:

| arm | cells | cpu lane | gpu lane (serialized) | **elapsed** |
|---|---:|---:|---:|---:|
| 1 · window 1000 | 143 | 16.4 cell-min → ~1.8 min | **82.1 min** | **~85 min** |
| 2 · `recondition=none` | 143 | 12.3 cell-min → ~1.3 min | 71.1 min | **~74 min** (upper bound) |
| 3 · one target | 117 | 10.9 cell-min → ~1.2 min | 63.1 min | **~65 min** each |
| 3 · all three targets | 351 | | | **~3 h 15 min** |
| **all five runs, in sequence** | **637** | | | **~6 h** |

**The projection checks itself.** Run over the primary grid's own cells it
gives 74 min against the 69.6 min that run actually took — 6% high, so these
are honest and slightly conservative.

Two adjustments downward, both already banked:

- **Arm 1 starts with its whole SPY column cached** — 13 cells, including the
  four GPU cells the probe computed (8.2 GPU-minutes). Remaining: ~77 min.
- **Arm 2's figure is an upper bound.** It is the primary's own cost, and
  `recondition=none` *skips* the `update` call at every non-refit origin,
  which is work the primary did. It cannot be slower than 74 minutes; the
  GPU lane, where the per-origin cost is the forward pass rather than the
  re-conditioning, will dominate either way.

Arms 2 and 3 need no window measurement: they score the primary's origins with
the primary's fits, and arm 3's only difference from the primary is which
column the loss is computed against.

## 7a. Disk

**19 GiB free on `/` (98% used), against a projected ~0.4 GiB for all five
runs.** Not a constraint, but the margin is thin enough to be worth stating: a
previous session found this filesystem at 100% with 4.7 G free.

| arm | fragments | projected | basis |
|---|---:|---:|---|
| 1 · window 1000 | 143 pairs | **~88 MiB** | measured: SPY's 13-cell column is 8.80 MiB against 9.77 at window 500, a ratio of 0.901 where the origin ratio is 0.898 — fragment size tracks origins exactly |
| 2 · `recondition=none` | 143 pairs | **~57–99 MiB** | same origins as the primary's 98.9 MiB, but the frozen forecast repeats for 20 origins out of 21 and parquet compresses it: measured, the 5-cell SPY sample is 2.26 MiB against the primary's own 3.88 MiB for the same five cells, a ratio of 0.58 |
| 3 · three targets | 351 pairs | **~263 MiB** | the primary's 117 equity cells (87.7 MiB) three times; measured, the 5-cell SPY sample is 3.88 MiB against the primary's 3.88 — identical row counts, only the proxy and loss columns differ |
| **total** | **637 pairs** | **~410 MiB** | |

The primary store is untouched by all of this: it stays at its 187 fragment
pairs, 130 MiB.

## 8. Carry this forward: the arms do not score the same origins

**Arm 1 is the one that has to be compared carefully, and this is the place to
say so rather than leaving it for whoever compares them later.**

At horizon 1 and step 1 a series of `n` observations has origins
`W-1 … n-2` under window `W`, so the window-1000 origin set is a *strict
subset* of the window-500 one: the same last origin, 500 fewer at the start.
Measured on SPY's stored fragments — 4,904 primary origins, 4,404 arm origins,
intersection 4,404, arm-only 0.

| asset | origins at 500 | at 1000 | coverage |
|---|---:|---:|---:|
| CAC | 5,038 | 4,538 | 90.1% |
| DAX | 4,996 | 4,496 | 90.0% |
| NDX | 4,942 | 4,442 | 89.9% |
| SPY, DIA | 4,904 | 4,404 | 89.8% |
| KOSPI | 4,837 | 4,337 | 89.7% |
| HSI | 4,828 | 4,328 | 89.6% |
| TWSE | 4,801 | 4,301 | 89.6% |
| NKX | 4,795 | 4,295 | 89.6% |
| BTC-USD, ETH-USD | 2,791 | 2,291 | 82.1% |

**The rule for any analysis that compares arm 1 with the primary: score the
comparison on the intersection of the origins both arms scored, or report the
two arms separately with their coverage stated.** A loss average, a Diebold–
Mariano statistic or a model-confidence set computed over each arm's own
origins is comparing two different samples as well as two different windows,
and on this panel the difference is 500 origins of 2005–2009 and 2017–2020 —
the highest-volatility stretch either series has. The intersection is exactly
the arm's own origin set, so the join is `origin_index`, and nothing has to be
interpolated or reindexed.

**And the crisis sub-samples are worse than a 10% haircut.** D-019 chose 500
on this measurement: at window 1000 the GFC window is largely inside the
warm-up — the panel holds 140–149 GFC days per equity series and an evaluation
at 1000 scores **31–86** of them — and BTC/ETH score **0 of their 90 COVID
days**, because 1000 days from the 2017 listing runs to 2020-05-13
(docs/PANEL_REPORT.md §8.1). So H3 cannot be tested on arm 1 for the crypto
arm at all, and its GFC reading is a fraction of the primary's. Arm 1 is a
*full-sample window-sensitivity* arm; it is not a second opinion on the crisis
results, and a crisis table computed from it would be a table about the
warm-up.

Arms 2 and 3 have no such problem: both score exactly the primary's origins
on exactly the primary's assets (§6), which is why arm 3's comparison is a
pure re-scoring and arm 2's is a pure protocol difference.

## 9. The launch commands

**Run them one at a time.** All three arms use the one GPU, and the GPU lane
is ~98% of a full run's elapsed time (docs/P3_GRID.md §3.1); two arms in
parallel would contend for the card and, at four TSFM cells' memory, can take
each other down. `run_grid` serializes the GPU lane *within* a run; nothing
serializes two runs.

Each is resumable: killing one and re-issuing the same line recomputes only
the cells missing from its store. Each already has its smoke cells cached
(§6), so the CPU lane starts five cells ahead — and, for arm 1, the four GPU
cells of the probe too.

From the repository root, with the Makefile's determinism exports (D-026's
kernel pin and D-032's thread pin; the driver refuses to start without them):

```bash
cd /home/martin/Documents/IJF/volbench/volbench

# arm 1 — window 1000, all 11 assets, 143 cells
NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nohup uv run --extra classical --extra tsfm \
  python -m volbench.benchmarks.grid_primary \
    --window 1000 \
    --tag window1000 \
    --out-dir data/grid_ablation_window1000 \
    --cpu-workers 12 \
  > data/grid_ablation_window1000/run_window1000.log 2>&1 &

# arm 2 — frozen forecast between refits, all 11 assets, 143 cells
NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nohup uv run --extra classical --extra tsfm \
  python -m volbench.benchmarks.grid_primary \
    --recondition none \
    --tag recondition_none \
    --out-dir data/grid_ablation_recondition_none \
    --cpu-workers 12 \
  > data/grid_ablation_recondition_none/run_recondition_none.log 2>&1 &

# arm 3 — the three robustness targets, 9 equity assets, 117 cells each.
# One nohup for all three: they share a store and must not overlap on the GPU.
NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nohup bash -c 'for T in parkinson garman_klass squared_return; do
  uv run --extra classical --extra tsfm \
    python -m volbench.benchmarks.grid_primary \
      --target "$T" \
      --assets NDX DAX CAC NKX HSI TWSE KOSPI SPY DIA \
      --tag "target_$T" \
      --out-dir data/grid_ablation_targets \
      --cpu-workers 12
done' > data/grid_ablation_targets/run_targets.log 2>&1 &
```

Notes on the shape, all of them checked rather than assumed:

- **The extras.** `--extra classical --extra tsfm` is what the GPU lane needs
  and what the primary grid ran under; `uv run --extra ...` syncs the
  environment to exactly the extras named, so naming only one would uninstall
  the other's backends. `uv sync --extra classical --extra tsfm --dry-run`
  reports **"Would make no changes"** against this checkout's `.venv`, so the
  line cannot move a package version — which matters more than usual here,
  because the TSFM adapters put their backend versions in `model.spec()` and
  therefore in the config hash. The four smoke runs of §6 were issued in the
  `--no-sync` shape for the same reason; either is safe on this box, and
  `--no-sync` is the one that cannot become unsafe.
- **`--cpu-workers 12`** is the primary run's fan-out on this 32-thread box
  (docs/P3_GRID.md §1). `--device` defaults to `cuda`.
- **No `--manifest-dir`.** Each run writes `docs/P3_GRID_manifest_<tag>.json`,
  a committed sibling that is not `docs/P3_GRID_manifest.json` and cannot
  become it: the archive-and-supersede path is keyed on the default tag.
- **Each `--out-dir` already exists** — the smoke runs of §6 created all three —
  so the log redirections above have somewhere to land. All are under
  gitignored `/data/`.
  The store lands in `<out-dir>/store`, the run's `summary_<tag>.json` and
  `report_<tag>.txt` beside it.
- **Watching progress**: `on_cell` fires as a lane's outcomes are *collected*,
  so a `ProcessExecutor` lane prints nothing until it finishes. Count
  fragments instead: `ls data/grid_ablation_*/store/*.parquet | wc -l` against
  the cell counts in §5.

## 10. Verification

| check | result |
|---|---|
| `ruff check .` | clean |
| `mypy` (`strict`, `src` + `test_model_interface.py`) | 56 source files, no issues |
| `pytest`, Python **3.11.5**, `--extra classical --extra tsfm` | **1,489 tests, 0 failed, 0 errors**, 29 skipped |
| `pytest`, Python **3.12**, `--extra classical --extra torch-cpu` | **1,489 tests, 0 failed, 0 errors**, 35 skipped |
| `pytest`, Python **3.13**, `--extra classical --extra torch-cpu` | **1,489 tests, 0 failed, 0 errors**, 35 skipped |
| `git ls-files -- data/` | empty |
| `tests/test_licensing_guard.py` | passed |
| `tests/test_identity_leakage.py` | passed |
| `tests/test_manifest_provenance.py` (M's counterpart guard, and its inert-proof) | passed |
| `tests/test_order_statistics.py` (O's column policy) | passed |
| manifests under `data/` without a committed counterpart | none |
| `manifest_provenance --check` on each new sibling manifest | both digests recompute |

**Resumability of the primary grid, which is what says the flags disturbed
nothing** (`--tag resume_after_p`, the default protocol, the study store):

| leg | result |
|---|---|
| the re-run | **143 cached, 0 computed, 0 failed** |
| its 143 config hashes vs the committed manifest's | identical set |
| its independently computed `store_digest` | `05efdb45…` — the committed manifest's |
| SHA-256 of all 374 store files, before vs after | identical |
| size + mtime of all 374 store files | identical — **not rewritten at all** |

Its manifest is archived at
`docs/archive/P3_GRID_manifest_resume_after_p.5f85971ddbbc.json` and the
working copy removed, as docs/P3_MANIFEST_INVENTORY.md §4 prescribes for a
verification run. The five smoke manifests are kept as live siblings under
`docs/`, because they are not verification artefacts: they name fragments the
arm runs will read back.

## 11. What this branch did not do

It did not run the arms. Nothing under `docs/` reports an ablation *result*,
and `docs/P3_GRID_manifest.json` is untouched.

Two things are recorded rather than acted on, per the standing rule that a
second change riding along is unattributable later:

- **`--refit-every` and `--step` are still literals.** D-034 schedules three
  arms and this branch exposes exactly those three. The refit cadence and the
  step are fields of the same `ProtocolArm` and would be one line each, but
  no arm asks for them and a flag with no measurement behind it is a flag
  nobody has checked the halves of.
- **`recondition` binds for models that cannot re-condition** (§4). It is the
  conservative direction and the store is unambiguous because of it, but it
  is the reason arm 2 recomputes 11 `patchtst` cells to prove they did not
  change. Worth a decision entry if a future arm makes that cost matter;
  not worth changing a hashing rule for.
