# Runner, parallel execution and economic value — volbench v0.5.0-runner

> Branch `feat/p2-runner`, off `main` at `ec27b50` (v0.4.0, tag
> `v0.4.0-protocol`). The last build item before the grid: orchestration,
> parallel execution, and the economic-value metric — plus the two
> model-protocol open questions that had to close before any grid freeze.
>
> **Where this file sits.** `docs/design.md` and `docs/decisions.md` are the
> durable records and both were updated. This is the branch's *evidence*: what
> was measured, what was gated, and the two things that were found rather than
> built. It is a new file rather than an append to `docs/P2_INTEGRATION.md`
> because that file is a planning-folder mirror and this task did not instruct
> an edit to it (CLAUDE.md); §7 below flags the drift it should absorb.

---

## 1. What shipped

| | |
|---|---|
| `src/volbench/runner.py` (new) | grid orchestration: `run_grid`, `GridSpec`, `ModelConfig`, `ProtocolArm`, `Cell`, `AssetData`, `DataSource`, `MappingDataSource`, `CellOutcome`, `RunManifest`, `read_grid_results` (D-027) |
| `src/volbench/execute.py` | `ProcessExecutor` — local multiprocessing backend, `kernel_signature`, the fork guard (D-028). `SerialExecutor` untouched. |
| `src/volbench/econ.py` (new) | `volatility_target_backtest`, `VolTargetBacktest`, `periods_per_year_for` (D-029) |
| `src/volbench/models/har.py` | HAR onto `models/_rv`, Duan smearing by default (D-030) |
| `src/volbench/models/patchtst.py` | `device_class` in `spec()`, `resolve_device_class` (D-031) |
| tests | `test_runner.py` (38), `test_econ.py` (37), `test_execute.py` (34, was 6), `test_models_har.py` (17, was 8), `test_models_patchtst.py` (+5) — 1159 in the suite, up from 1073 |
| docs | `docs/design.md` as-built; `docs/decisions.md` D-027…D-031; this file |

Version **0.4.0 → 0.5.0**. Required, not cosmetic: D-030 changes what HAR
computes and D-031 changes what a PatchTST cell *is*, so scored-path numbers
move and `package_version` is in every config hash. Every pre-0.5.0 fragment
is orphaned — never overwritten, never served — which is the mechanism working
as D-021 described it. Tag `v0.5.0-runner`.

---

## 2. D-030 — HAR's retransformation, measured before and after

The brief asked for the effect to be measured rather than assumed. Toy
fixture, 200 origins, window 500, refit every origin, scored against
`overnight_plus_range` (D-016), compared against the fixture's **known**
`true_variance` (`make_toy_asset` records it, which is what makes this a
calibration measurement and not a comparison of two estimators):

| HAR arm | model name | forecast/true | QLIKE vs **true** | QLIKE vs proxy | hit @5% |
|---|---|--:|--:|--:|--:|
| Gaussian `exp(ŷ + ½σ²)` — **before** | `har_rv` | **1.1320** | **0.0263** | 0.1823 | 0.035 |
| Duan smearing — **after** (default) | `har_rv-smearing` | **1.1102** | **0.0242** | 0.1806 | 0.035 |
| Gaussian, kept as the arm | `har_rv-gaussian` | 1.1320 | 0.0263 | 0.1823 | 0.035 |

Three things to read off it:

1. **The `gaussian` arm reproduces the pre-0.5.0 numbers to every printed
   digit.** That is what makes the first two rows a measurement of the
   retransformation rather than of an accidental refactor.
2. **Smearing removes about a sixth of the overshoot**, not all of it. Reported
   as such. The residual is HAR's own *one-step, in-sample* factor, which is
   the deeper problem docs/M2_NOTES.md named; a component overnight+intraday
   HAR remains open (design.md).
3. **The toy is a toy.** Its overnight share is a generator parameter, not a
   real index's. The direction is what the M2 analysis predicted; the
   magnitude is this fixture's.

Toy benchmark effect (8 models, `make reproduce`): HAR's CRPS 0.005860 →
0.005857, QLIKE 0.1823 → 0.1806, mean σ̂ 0.01154 → 0.011427. The other seven
models' scores are **unchanged to every printed digit** — each cell is
independent, and only HAR's changed. All eight config hashes moved, with the
version.

---

## 3. D-031 — PatchTST's device class, in the hash

```
device    device_class   sha256(spec)[:16]
cpu       cpu            7be4c7bf5532ba0c
cuda      cuda           eeb90f2f72f5bc5a
cuda:0    cuda           eeb90f2f72f5bc5a
cuda:3    cuda           eeb90f2f72f5bc5a
auto      cuda           eeb90f2f72f5bc5a     (on the 4090 box)
```

The class splits the hash; the ordinal does not. `make smoke-tsfm` writes
`device_class: cuda` into the stored config sidecar, so a fragment now records
which class produced it. PatchTST's *numbers* are unchanged (CRPS 0.005856,
QLIKE 0.182143 — identical to the v0.3.0 record); only its identity moved.

The zero-shot TSFM adapters keep `device` out of `spec()` deliberately: they
estimate nothing, so no RNG stream is drawn from, and a forward pass differs
across devices only by float accumulation order — the tolerance question D-026
already scopes.

---

## 4. The gate that matters — serial vs parallel byte-identity

`tests/test_runner.py::TestSerialParallelIdentity` runs one grid through both
backends into two stores and compares the parquet **bytes** fragment by
fragment, with an inert-proof companion (perturb one input by one ulp; the
comparison must notice). Green.

Measured again at grid scale, one process per measurement so no run pollutes
the next — 32 cells (4 synthetic assets × the toy benchmark's 8 models), window
500, h=1. The right-hand column is a sha256 over the concatenated fragments in
config-hash order:

| backend | refit 21 | refit 1 | fragments (refit 1) |
|---|--:|--:|---|
| `SerialExecutor` | 10.16 s | 121.75 s | `e4c3c0d2ce7210f8` |
| `ProcessExecutor(4, forkserver)` | 6.43 s | 45.48 s | `e4c3c0d2ce7210f8` |
| `ProcessExecutor(8, forkserver)` | 6.65 s | **31.99 s** | `e4c3c0d2ce7210f8` |
| `ProcessExecutor(16, forkserver)` | 7.52 s | 34.70 s | `e4c3c0d2ce7210f8` |
| `ProcessExecutor(8, fork)` | 6.23 s | 31.62 s | `e4c3c0d2ce7210f8` |

**One fingerprint across every backend, worker count and start method.** That
is D-011's H4 identity claim, verified at grid scale rather than asserted.

On wall-clock, stated honestly rather than flattered:

- At `refit_every=1` (cells of ~3.8 s) the pool reaches **3.8× at 8 workers**
  and gets no better at 16. H4 asks for ≥5× on 8 cores; **this measurement
  does not reach it**, and the reason is visible in §6: each worker's own BLAS
  is multi-threaded, so 8 workers on 32 logical cores are already
  oversubscribed. Single-threading the workers gets to 4.7×–5.1× — and is
  *rejected*, because it changes GARCH's numbers (§6). The honest reading is
  that H4's engineering claim needs a machine-level thread budget decided
  alongside the reproducibility decision, not a runner change.
- At `refit_every=21` (cells of ~0.3 s) the speedup is only 1.6×: pool startup
  dominates. The study's grid is the first regime, not this one.
- Resuming a complete 32-cell grid: **0.30 s**, 32/32 cached, nothing refitted.

---

## 5. Found while measuring — `fork` deadlocks after LightGBM

The first grid benchmark hung. Bisected to a single cause, reproducible:

```
parent runs one grid serially (trains LightGBM) → forks a pool → worker's
first LightGBM call never returns
```

| start method | outcome |
|---|---|
| `fork` | **deadlock** (killed at 90 s; reproduced every time) |
| `fork`, with `OMP_NUM_THREADS=1` set before the parent starts | 0.19 s ✓ |
| `forkserver` | 1.22 s ✓ |
| `spawn` | 1.49 s ✓ |

LightGBM is OpenMP-backed; `fork` copies the runtime's locks in whatever state
the parent's threads left them, and the child's first parallel region waits on
a lock no thread will release. Per model, with the parent having run the same
model serially first: naive, EWMA, GARCH, HAR, AutoETS, AutoARIMA all fork
fine; only LightGBM hangs.

Consequences, both taken (D-028):

- **The default start method is `forkserver`**, not `fork`. Workers are forked
  from a clean server process, so nothing the parent did to a native runtime
  reaches them. Cost measured at 1% of grid wall-clock (§4).
- **`fork` is refused where it would hang** rather than left to deadlock:
  `map` raises if a module in `FORK_UNSAFE_MODULES` is in `sys.modules`. That
  test is exact rather than conservative because every optional backend is
  imported inside `fit` and nowhere else (D-022) — so "imported" means "already
  used in this process".

There is deliberately **no test that reproduces the deadlock**; a hanging test
is worse than the bug. `TestForkIsRefusedWhereItWouldHang` pins the guard.

The new default has one cost, and it is also turned into a message: every start
method other than `fork` re-imports the parent's `__main__` in each worker, so
a REPL, `python -` or a heredoc cannot host a pool. Unguarded that arrives as
`BrokenProcessPool` with a `FileNotFoundError: '<stdin>'` buried in a worker's
traceback; `_require_importable_main` refuses it up front and names both ways
out (a script with the usual `if __name__ == "__main__":` guard, or `fork`).
The README's grid snippet is run verbatim as a script as part of this branch's
checks, so the documented usage is the tested one.

---

## 6. Found while measuring — the OpenBLAS thread count changes GARCH's numbers

**Not introduced by this branch, not fixed by it, and the more serious of the
two findings.** Reported here in full because it is a reproducibility defect
that affects the paper's numbers.

Toy benchmark, 8 models, one machine, varying only the BLAS thread
environment. `forecast_var` compared row for row:

| comparison | result |
|---|---|
| default threads, run twice | identical, all 8 models |
| `OPENBLAS_NUM_THREADS=1`, run twice | identical, all 8 models |
| default vs `OMP+OPENBLAS+MKL=1` | **differs: `garch11`, `garch11_t`** — six models identical |
| default vs `OPENBLAS_NUM_THREADS=1` alone | **same difference** — so it is OpenBLAS, not OpenMP generally |
| `OPENBLAS_NUM_THREADS=1` vs all-three-set | identical |

Magnitude: max relative difference in `forecast_var` **9.2e-5** for `garch11`,
**5.5e-1** for `garch11_t`. Mechanism: threaded BLAS reorders a reduction by an
ulp, and `arch`'s SLSQP amplifies it into a different local optimum — a large
move on a few origins, not a uniform rounding difference.

Why it matters more than D-026's case: D-026's kernel-family difference moves
the *content digest* of a computed proxy, so the store **misses** and
recomputes. This one moves **results without moving the config hash**, so the
store will happily serve a 32-thread GARCH fragment for a 2-core run's request.
A grid filled on this box and topped up on a CI-sized machine would hold two
answers under one hash.

What is *not* wrong today: two runs at the same thread count are bit-identical;
serial and pooled runs agree exactly, because workers inherit the parent's
environment (this is why §4's fingerprints match and why a `worker_threads=1`
option was rejected — it would have made the pool stop reproducing the serial
backend, the one thing the execute seam must never do).

**Recommendation, not taken here.** Pin `OPENBLAS_NUM_THREADS` in the Makefile
and CI exactly as D-026 pins `NPY_DISABLE_CPU_FEATURES`, and restate the
determinism rule as "same seed, same code, same data, same kernel family,
**same BLAS thread count**". That moves every GARCH number, so it is a protocol
decision with its own entry, for the planning machine to number — this branch's
five decision numbers were reserved by its brief. It should be taken **before**
any grid freeze, for exactly the reason the brief gave for D-030.

---

## 7. Drift flagged, not edited

- `docs/P2_INTEGRATION.md` §3.4 ("`device` is not hashed") and §6.6 ("what
  'the same model' means across devices — OPEN") are superseded by D-031;
  §6.7 ("HAR is now the odd one out — OPEN") is superseded by D-030. The file
  is a mirror and this task instructed no edit to it.
- `docs/research_design.md` H4 says "≥5× on 8 cores"; §4 measures 3.8× and
  explains why. The design doc is a mirror and is not edited; the number is
  reported here.
- `docs/M2_NOTES.md`'s "HAR's lognormal retransformation is sensitive to the
  target's log-space noise… a Phase-2 modelling item" is now D-030. CLAUDE.md
  permits updating that mirror only when a task instructs it; this one did
  not, so it is flagged rather than edited.

---

## 8. Gates

### 8.1 Local

| gate | result |
|---|---|
| `uv run ruff check .` | clean |
| `uv run mypy` (strict, `src` + `test_model_interface.py`) | clean, 43 source files |
| `uv run pytest` — Python 3.11 | **1159 passed, 29 skipped**, 461 s |
| Python 3.12 (`UV_PROJECT_ENVIRONMENT=.venv-py3.12`) | **1159 passed, 35 skipped**, 481 s |
| Python 3.13 (`UV_PROJECT_ENVIRONMENT=.venv-py3.13`) | **1159 passed, 35 skipped**, 473 s |
| `pytest -m "tsfm or gpu"` on the 4090 (`VOLBENCH_RUN_TSFM=1 VOLBENCH_RUN_GPU=1`, `--extra tsfm`) | **28 passed, 0 skipped**, 17 s |
| `make benchmark` twice, `diff -r` | **byte-identical** across two full rebuilds |
| `make smoke-tsfm` twice, `diff -r` | **byte-identical** across two full runs |
| serial vs `ProcessExecutor` identity | green in the suite, and §4 at grid scale |

The 3.12/3.13 legs skip six more tests than 3.11 because those environments
carry no torch (the PatchTST CPU smoke tests `importorskip`); CI installs
`torch-cpu` on every leg and runs them.

### 8.2 Toy benchmark at 0.5.0

```
label       model                   CRPS      log score   QLIKE    mean σ̂
ewma        ewma                    0.005820  -3.185884   0.161594  0.010781
lgbm        lightgbm_rv-smearing    0.005838  -3.172569   0.175685  0.011223
autoets     autoets_rv-smearing     0.005847  -3.168580   0.181821  0.011308
autoarima   autoarima_rv-smearing   0.005849  -3.167422   0.181412  0.011288
har         har_rv-smearing         0.005857  -3.161706   0.180571  0.011427
garch11_t   garch(1,1)-studentst    0.005872  -3.159554   0.172877  0.011882
garch11     garch(1,1)-normal       0.005873  -3.159685   0.172858  0.011871
naive       naive_rw_vol            0.006013  -3.078446   0.302762  0.013536
```

Pinned identities (`tests/test_recondition.py`), all eight moved with the
version and HAR's with D-030 as well:

```
autoarima 61a64971…65322e26   autoets 8b9eb9fc…d5c4f8280   ewma  5cd9e3e1…e507a823f73c7
garch11   c708c06e…d44b991b   garch11_t 7a4f4358…84944de64f  har  bf3283ac…2ad6c80c947
lgbm      5628598d…262957933   naive  da5fb0f6…4cb0373901a
```

`make smoke-tsfm` on the 4090 (4 models, refit 21): timesfm 0.005841 / chronos
0.005852 / moirai 0.005853 / patchtst 0.005856 CRPS — unchanged from the
v0.3.0 record; four new hashes with the version, PatchTST's also with D-031.

### 8.3 CI

`.github/workflows/ci.yml` on the branch push — ubuntu-latest × Python
3.11/3.12/3.13, `--extra classical --extra torch-cpu`, ruff + mypy + pytest,
with `NPY_DISABLE_CPU_FEATURES` pinned per D-026. Green on the branch
(run 32856459336, 17m35s, and the two follow-up pushes).

### 8.4 Leakage audit

Full-diff audit against `.claude/skills/leakage-check`, with the econ
position/return alignment as the focus: see §9.

---

## 9. Leakage check — `main...feat/p2-runner`

| # | Item | Verdict |
|---|---|---|
| 1 | Index arithmetic at boundaries | **PASS** — no new index arithmetic anywhere in the diff. `runner.Cell.splitter()` constructs a `RollingOriginSplitter` and hands it to `run_backtest`; `econ.py` does no indexing at all beyond selecting one horizon and sorting by `origin_index`. |
| 2 | Splitter monopoly | **PASS** — `ProtocolArm.splitter(horizon)` is the only place a splitter is built, and it builds a `RollingOriginSplitter`. `grep` over the diff finds no `.iloc[` window, no date filter and no hand-rolled slice in `runner.py`/`econ.py`/`execute.py`. `tests/test_runner.py::test_the_splitter_is_the_arms_and_nothing_is_hand_rolled` pins the parameters through. |
| 3 | Feature lags | **PASS** — no new features. HAR's design matrix is unchanged by D-030 (only the factor applied to its output moved); `tests/test_models_har.py::test_each_arm_applies_exactly_its_own_factor` shows the two arms share `beta` exactly. |
| 4 | Transforms and scalers | **PASS** — both HAR retransformation factors are estimated from the *fit window's own in-sample residuals*, once, at the scheduled fit, and `update` re-estimates neither (pinned). No scaler is introduced anywhere in the diff. |
| 5 | Target construction | **PASS** — untouched. `econ.py` reads `realized_return` as stored and never rebuilds a target. |
| 6 | Refit schedule | **PASS** — `ProtocolArm` carries `refit_every` into the splitter and `recondition` into `run_backtest`; the runner adds no off-schedule refit. A failed scheduled fit still fails its whole block (evaluator behaviour, unchanged). |
| 7 | TSFM context windows | **PASS** — unchanged. The runner does **not** batch across assets; the GPU lane serializes whole cells, so no cross-series padding or alignment is introduced. Batching stays open and still needs its own audit when built. |
| 8 | Calendar alignment | **PASS** — `AssetData` carries returns, proxy and variance as one bundle and `run_backtest` re-checks their indexes are identical and ascending. Nothing in the runner reindexes or joins across assets; each cell sees exactly one asset. `econ.py` never joins anything. |
| 9 | Caching | **PASS**, and strengthened. Resume is by `config_hash`, which already covers the data's content digests; `tests/test_runner.py::TestResumability` checks fragments are byte-identical *and unrewritten* (mtimes) after a resume. D-031 closes a real hole in this item: a PatchTST CPU fragment could previously be served for a GPU request under one hash. **Flagged, not fixed:** the OpenBLAS thread count moves GARCH results without moving the hash (§6) — item 9's failure mode exactly, at machine level rather than at cache level. |
| 10 | Survivorship & selection | **PASS** — the grid is declared, not selected on outcomes. `GridSpec` is expanded in full and every cell is attempted; a failed cell is recorded, never dropped, and `n_missing` keeps the evaluator's per-origin NaNs visible from the manifest, so a model cannot look good by averaging over the cells that happened to work. |

### The focus item — the economic-value alignment

`position_t = min(target_vol / forecast_vol_t, cap)` applied to the realized
return of `t+1`. In the stored schema a row *is* the pair
`(forecast issued at origin_index, realization at target_index = origin_index +
horizon)`, and the splitter guarantees `target_index > origin_index`. So the
backtest is one element-wise product over rows and there is no lag, shift or
join for an off-by-one to hide in. Three independent pins:

1. `test_the_arithmetic_is_position_from_this_row_times_this_rows_return` —
   hand-computed positions and P&L on three rows, no statistics involved.
2. `test_shifting_the_returns_by_one_day_changes_the_sharpe_materially` —
   the requested canary. Shifting `realized_return` one row in either
   direction must drop the Sharpe below 75% of the aligned value, on four
   seeds. Measured ratios 0.42–0.59, so the test has margin and can fail.
   `test_only_the_aligned_run_actually_hits_its_volatility_target` adds the
   sharper symptom: aligned realized vol lands within 5% of the 10% target,
   misaligned is >3× off.
3. `test_no_column_is_shifted_lagged_or_reindexed_inside_the_module` — a
   structural check that `econ.py` contains no `.shift(`, `.reindex(` or
   `np.roll(`, so no second opinion about alignment can appear later.

**A bound on what (2) proves, reported because it matters.** The damage a
one-day shift does is governed by the dispersion of `σ_{t+1}/σ_t`. On a
*strongly persistent* volatility series that ratio is near 1 and the misaligned
Sharpe is within a few percent of the aligned one —
`test_the_shift_is_hardest_to_see_exactly_where_volatility_is_persistent`
measures exactly that and asserts it. So on real data no statistical check
would catch this bug; the structural pins (1) and (3) are what actually hold
the line, and the canary is calibrated on a deliberately dispersed series to
have any power at all.

**Direction, checked explicitly.** Sizing today's position with *tomorrow's*
forecast is the classic vol-targeting look-ahead and flatters Sharpe. Here
that would require reading `forecast_var` from a different row than
`realized_return`, which the one-product implementation makes impossible.

### The demanded canary, at grid level

`tests/test_runner.py::TestLeakageCanary` — corrupt every observation strictly
after a cutoff T, run the same grid again, and require every forecast whose
target lands at or before T to be **bit-identical**. It runs under **both**
backends, which makes it two claims at once: that the runner introduced no
index arithmetic, and that the pool pooled no state across cells. Its
inert-proof companion corrupts the *past* and requires the same comparison to
fail. `tests/test_m1_smoke.py`'s benchmark-level canary is unchanged and still
green.

### Verdict

No FIX and no FATAL findings in the diff. One pre-existing item is escalated:
the OpenBLAS thread sensitivity (§6) is a genuine item-9 hazard — same hash,
two answers — and it should be closed before the grid freeze.
