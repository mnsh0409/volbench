# The BLAS thread count, the fallback, and `nu` — volbench v0.6.0-determinism

> Branch `feat/p3-determinism`, off `main` at `0bedaed` (v0.5.0, tag
> `v0.5.0-runner`). It closes the reproducibility defect
> `docs/P3_RUNNER.md` §7 reported and could not take, and it does so somewhere
> §7 did not expect.
>
> **Where this file sits.** `docs/design.md` and `docs/decisions.md` are the
> durable records and both were updated (D-032). This is the branch's
> *evidence*: what was measured, what the measurement contradicted, and the
> one remedy that was measured and then refused. A new file rather than an
> edit to `docs/P3_RUNNER.md`, which is another branch's evidence record; §8
> below says what it supersedes there.

---

## 1. What shipped

| | |
|---|---|
| `src/volbench/determinism.py` (new) | `kernel_signature` (moved from `execute.py`, re-exported), `thread_pin`, `is_pinned`, `determinism_env`, `blas_info`, `environment_spec`, `environment_report`, `KERNEL_PIN_VAR`, `THREAD_PIN_VARS`, `PINNED_THREADS` |
| `src/volbench/results.py` | `build_config` records an `environment` block; `blas_threads` is in every config hash (D-032) |
| `src/volbench/execute.py` | both pins propagated to workers; each worker checks its own thread count against the parent's |
| `src/volbench/models/base.py` | `FitDiagnostics`, `SupportsFitDiagnostics` — the optional "how did the fit go" protocol |
| `src/volbench/models/garch.py` | `fit_diagnostics()`; `nu` bounded to `NU_BOUNDS = (2.1, 50)`; `FIT_TOL = 1e-10`; both in `spec()` |
| `src/volbench/evaluate.py` | `fit_status` on every scored row |
| `src/volbench/runner.py` | `n_fits` / `n_fits_fallback` / `n_fits_nonconverged` per cell; `environment` on the manifest |
| `Makefile`, `.github/workflows/ci.yml` | `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` |
| tests | `test_determinism.py` (28, new), plus additions to `test_models_garch.py`, `test_evaluate.py`, `test_runner.py`, `test_execute.py`, `conftest.py` — **1153 passing**, up from 1104 |
| docs | `docs/design.md` as-built; `docs/decisions.md` D-032; this file |

Version **0.5.0 → 0.6.0**. Required: `environment` moves every config hash, and
`nu_bounds` moves the GARCH-t *numbers*. Tag `v0.6.0-determinism`.

Machine for every measurement below: the 4090 box (D-010), 32 logical cores,
numpy 2.4.6 on scipy-openblas **0.3.31.188.0** (`USE64BITINT DYNAMIC_ARCH
NO_AFFINITY Haswell MAX_THREADS=64`, pthreads), `NPY_DISABLE_CPU_FEATURES` at
its Makefile value throughout. "default threads" means 32.

---

## 2. The diagnosis — the hypothesis was wrong, and the real cause was next to it

The brief's hypothesis: the 5.5e-1 is not float reordering, it is `GARCH.fit`'s
EWMA fallback firing on some origins under one thread count and not the other.

**It is not.** The toy `garch11_t` cell was run at both thread counts with the
fallback flag and `arch`'s own optimizer report recorded per origin:

| | default (32 threads) | `OPENBLAS/OMP=1` |
|---|--:|--:|
| origins | 200 | 200 |
| **fallbacks fired** | **0** | **0** |
| **`convergence_flag != 0`** | **0** | **0** |
| origins whose `forecast_var` differs at all | 200 | — |
| … by more than 1e-3 relative | 122 | — |
| … by more than 1e-1 relative | 2 | — |
| max relative difference | **5.531e-01** | — |

So the count of origins where the fallback fired under one thread count and not
the other is **zero**, and the count of origins that differ materially is 122.
The two sets are disjoint by construction. The 5.531e-01 reproduces §7's
5.5e-1 exactly, at origin 568.

**What is actually happening.** `nu` is not identified on a 500-observation
window of this series:

| | default | `=1` |
|---|--:|--:|
| estimated `nu`, range over 200 origins | [109.2, **500.0**] | [110.8, **500.0**] |
| max &#124;Δ`nu`&#124; between the two runs | **377.4** | |
| max &#124;Δ log-likelihood&#124; | **2.06e-01** | |
| mean &#124;Δ log-likelihood&#124; | 6.04e-02 | |

500 is `arch`'s own upper bound on `nu`. The likelihood is flat up there — at
`nu` = 50 a Student-t already matches a Gaussian past float precision in every
quantile this project scores — so SLSQP stops wherever it drifts to. The
log-likelihood differences settle it: these are **different local optima**, not
one optimum found to different precision.

The flat direction couples into the variance parameters. On 10 origins
(default) / 12 (`=1`) the optimizer lands at `alpha ≈ 0, beta ≈ 0.995`, an
IGARCH-like corner; on **2** origins the two thread counts disagree about
whether to take it, and those two are exactly the rel > 1e-1 origins. The worst:

```
origin 568   forecast_var  1.577e-04 -> 2.450e-04   (5.53e-01 relative)
             alpha         2.675e-02 -> 4.547e-22
             beta          0.9664    -> 0.9946
             loglik     -839.7535    -> -839.7971
```

**The control that closes it.** The same cell with normal innovations — same
data, same optimizer, same alpha-corner phenomenon (10 vs 8 origins at the
corner, 2 disagreements) — moves by **9.165e-05**, which is §7's 9.2e-5:

| | garch11_t | garch11 (normal) |
|---|--:|--:|
| bit-identical origins | 0/200 | 56/200 |
| max relative difference | 5.531e-01 | 9.165e-05 |
| origins > 1e-3 relative | 122 | 0 |
| max &#124;Δ log-likelihood&#124; | 2.06e-01 | 6.79e-05 |

Three orders of magnitude in the likelihood gap. Without a flat direction the
same ulp never switches a basin. The generator uses `rng.standard_normal()`, so
the toy series' true `nu` is infinite and the Student-t model is misspecified
in precisely the direction that makes `nu` unidentifiable.

**The answer, stated either way as the brief asked.** The fallback mechanism is
**not** what causes the thread sensitivity: 0 origins, both thread counts, both
the toy fixture and the real panel (§5). The cause is a weakly identified `nu`
and the flat likelihood it produces — which is the parenthetical the brief
attached to item 3, arriving through optimum-hopping rather than through the
fallback.

---

## 3. Fallback and convergence, made visible regardless

Done independently of §2's answer, because the brief asked for it either way —
and because §2 is the reason it is worth having: "the fallback did not fire" is
a claim somebody has to be able to check without an instrumented run.

**Per row.** `fit_status` on every scored row, from an optional
`FittedModel.fit_diagnostics()`. `""` for a model that reports nothing and for
rows with no fit behind them; `ok|flag=0 nu=7.09`; `fallback=ewma|flag=9`. The
empty string is reserved for "did not say", so it can never be read as "did not
fall back". It describes the fit at `fit_origin`, not the origin — `update`
re-conditions at fixed parameters and runs no optimizer — which is what makes
"this cell fell back on N of its M fits" answerable from the stored rows.

**Per cell.** `n_fits`, `n_fits_fallback`, `n_fits_nonconverged` on every
`CellOutcome`, counted per *scheduled fit* rather than per row: a block of 21
origins rests on one fit and a cell at horizon 5 writes five rows per origin,
so a row-weighted rate would really be a statistic about the refit cadence.
`fallback_rate` is `nan`, never `0.0`, when nothing reported.

**Per run.** `RunManifest.environment` — `blas_threads`, whether the pin is
explicit, the kernel signature, the CPU count, the pins as they stood, the BLAS
name/version/configuration, and the observed thread pools.

The schema was widened rather than routed to the manifest only: the row-level
column is what makes the per-cell counts checkable against the data, and
`tests/test_runner.py::TestFitCountsReachTheManifest` asserts the manifest
count *is* what the fragment says rather than recomputing it the same way
twice.

---

## 4. The source fix — measured, including the remedy that was refused

The brief made item 3 conditional on §2 confirming the fallback mechanism. It
did not. The remedies were implemented and measured anyway, because §2 confirms
the *root cause* item 3 names — a weakly identified `nu` and a flat likelihood
— and because a pin that freezes a knife-edge estimator is a worse outcome than
a pin over a stable one. All four variants, toy `garch11_t`, default threads vs
`=1`, 200 origins:

| variant | max rel. diff | origins > 1e-3 | origins > 1e-2 | max &#124;Δ loglik&#124; | non-conv |
|---|--:|--:|--:|--:|--:|
| as shipped at 0.5.0 | 5.531e-01 | 122 | 13 | 2.06e-01 | 0 |
| `ftol=1e-10` only | 5.538e-01 | 88 | 1 | 2.06e-01 | 0 |
| `nu <= 50` only | 2.269e-04 | 0 | 0 | 5.82e-07 | 0 |
| **`nu <= 50` + `ftol=1e-10`** | **2.927e-06** | **0** | **0** | **3.86e-10** | **0** |
| + multi-start, argmax loglik | 3.285e-01 | 1 | 1 | 1.94e-02 | 0 |
| + multi-start, margin loglik | 3.285e-01 | 1 | 1 | 1.94e-02 | 0 |

Read off it:

1. **The tolerance alone does nothing** (5.54e-01). The optimizer was
   converging; tightening where it stops does not help when the surface has two
   optima. Reported because it is the negative result that says the problem was
   never precision. Note this is a *tighter* tolerance, never a widened one —
   the failure the brief warned against would have been the opposite move, and
   the non-convergence column is zero throughout, so nothing was made to
   "pass".
2. **The `nu` bound is the fix** — 5.5e-1 → 2.3e-4, and 2.9e-6 with the
   tolerance on top. Adopted: `NU_BOUNDS = (2.1, 50)`, `FIT_TOL = 1e-10`, both
   in `spec()` and therefore in the hash. **189,000×** on the headline number,
   and `garch11_t`'s residual thread sensitivity (2.9e-6) is now *below*
   `garch11`'s own (9.2e-5) — the pathological amplification is gone and only
   ordinary ulp propagation is left.
3. **Multi-start is measured and rejected.** Taking a max over twelve
   independently thread-sensitive optimizations is more variable than running
   one. At origin 567 the restarts find a genuinely better optimum (+0.0194
   nats) under `=1` and not under 32 threads, so the multi-start converts a
   deterministic-but-suboptimal answer into a thread-dependent one. A
   margin-based selection rule (accept a challenger only on a relative
   log-likelihood improvement) does not rescue it, because the winning
   challenger clears any sane margin — the lottery is over which restart
   escapes, not over ties. It found a better optimum on **1 of 200** origins,
   for **12×** the fits. Not shipped; recorded here so it is not re-proposed.

**The bound is not a Gaussian in disguise.** On the toy fixture `nu` lands at
exactly 50 on 200/200 origins — the honest reading of "the true `nu` is
infinite here". On real panel data it is interior and estimable:

| cell | `nu` min | median | max | at the 50 bound |
|---|--:|--:|--:|--:|
| DAX/garch11_t | 3.32 | 7.09 | 38.71 | 0/238 (0.0%) |
| KOSPI/garch11_t | 3.82 | 7.97 | 50.00 | 12/231 (5.2%) |
| NDX/garch11_t | 3.56 | 7.39 | 50.00 | 6/236 (2.5%) |

18 of 705 fits (2.6%) at the bound. The bound removes the unidentified region
and leaves the estimable one alone — which is the argument for choosing 50,
made before the sensitivity was measured rather than after.

**What moved on the toy benchmark** (`make benchmark`, refit_every=1):
`garch11_t` QLIKE 0.172877 → 0.174629 and mean forecast vol 0.011882 →
0.011950; `garch11` QLIKE 0.172858 → 0.172856 (the tolerance alone); the other
six models are unchanged to every printed digit. No number in that table means
anything about a model — the series is synthetic — but the *pattern* is the
check that the change reached what it was supposed to and nothing else.

---

## 5. The fallback rate on real panel cells

Read off a run manifest, not an instrumented run — which is the point of §3.
Three real assets from the D-004 panel, both GARCH configurations, headline arm
(window 500, refit every 21, daily re-conditioning), h=1:

```
cell                       fits  fallback     rate  nonconv    rows
DAX/garch11                 238         0   0.0000        0    4996
DAX/garch11_t               238         0   0.0000        0    4996
KOSPI/garch11               231         0   0.0000        0    4837
KOSPI/garch11_t             231         0   0.0000        0    4837
NDX/garch11                 236         0   0.0000        0    4942
NDX/garch11_t               236         0   0.0000        0    4942

grid total: 0/1410 fits fell back        blas_threads=1 explicit=True
```

**0.00% on 1410 scheduled fits**, before and after the §4 change alike. The
fallback is a guard that does not fire on this panel, and that is now a
measured statement rather than an assumption. It is exactly why §2's answer had
to come first: had the rate been non-zero, the fix in §4 would have been the
wrong fix.

The rate stays worth carrying. It is measured on three of eleven assets, at one
arm; a crisis sub-sample or a shorter window is where a GARCH-t stops
converging, and `docs/design.md` now carries the open question of what a study
*does* with a cell above some rate.

---

## 6. The pin, and what it costs

`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`, exported by the Makefile
(`?=`, so a deliberate measurement can override) and set job-level in CI.
Exported for the whole target, **not** injected into workers only: a pool
inherits the parent's environment, and pinning workers alone is precisely how a
pooled run stops reproducing a serial one — the failure the execute seam exists
to prevent.

The pin is a discipline. `blas_threads` in the config hash is the mechanism:

- Pinned everywhere, every machine records `1`, and cross-machine cache sharing
  (D-011) is unaffected.
- Unpinned on a 32-core box vs an 8-core runner, the two record 32 and 8, get
  different hashes, and the store **misses**. That converts the pre-D-032
  failure — one hash, two answers, either served for the other — into D-026's
  strictly milder one: a wasted recomputation.

The hashed value is the *resolved* count (`OPENBLAS_NUM_THREADS`, else
`OMP_NUM_THREADS`, else `os.cpu_count()`), deliberately not a runtime
introspection of the loaded BLAS: a hash may not depend on whether an optional
introspection package is installed. `threadpoolctl`, when present, feeds only
the manifest.

Cost, stated plainly: every pre-0.6.0 fragment is orphaned, and the committed
toy identities are now only *defined* under the pin. Tests that assert one
carry `@pytest.mark.pinned_identity` and **skip** on an unpinned shell with a
message naming the pin, rather than reporting a machine's different-but-correct
answer as a regression. `make check` and CI always run pinned.

---

## 7. Gates

### 7.1 Local

```
uv run --extra classical pytest      1153 passed, 55 skipped (7:13)
uv run ruff check .                  All checks passed
uv run mypy                          Success: no issues found in 44 source files
make benchmark                       fixture regeneration a no-op; 8 identities as pinned
```

### 7.2 Serial vs parallel byte-identity, re-verified under the pin

Redoes §4 of `docs/P3_RUNNER.md` with the pin in force. 32 cells (4 synthetic
assets × the toy benchmark's 8 models), window 500, h=1, one process per row.
Right-hand column is a sha256 over the concatenated fragments in config-hash
order.

| backend | refit 21 | fingerprint | refit 1 | fingerprint |
|---|--:|---|--:|---|
| `SerialExecutor` | 8.80 s | `8fe03d0b1946b852` | 115.80 s | `f7d34c7215c2f7fb` |
| `ProcessExecutor(4, forkserver)` | 3.32 s | `8fe03d0b1946b852` | 34.53 s | `f7d34c7215c2f7fb` |
| `ProcessExecutor(8, forkserver)` | 2.53 s | `8fe03d0b1946b852` | **24.14 s** | `f7d34c7215c2f7fb` |
| `ProcessExecutor(16, forkserver)` | 2.33 s | `8fe03d0b1946b852` | 22.81 s | `f7d34c7215c2f7fb` |
| `ProcessExecutor(8, fork)` | 1.68 s | `8fe03d0b1946b852` | 23.78 s | `f7d34c7215c2f7fb` |

**One fingerprint per refit cadence across every backend, worker count and
start method.** D-011's identity claim survives the pin, which was the thing
that had to be checked: a pin applied to workers only would have broken it.

### 7.3 H4's speedup, re-measured under the pin

At `refit_every=1`, against the serial baseline in the same table:

| workers | wall | speedup | before the pin (P3_RUNNER §4) |
|--:|--:|--:|--:|
| 4 | 34.53 s | 3.35× | — |
| **8** | **24.14 s** | **4.80×** | **3.8×** |
| 16 | 22.81 s | 5.08× | ~3.8× (no better at 16) |

**4.80× on 8 workers, up from 3.8×.** H4 asks for ≥5× on 8 cores and this is
still short of it, by 4%. Reported as short rather than rounded: the target is
met at 16 workers (5.08×), and 16 workers is not 8 cores. What the pin did fix
is the *reason* §4 gave for the gap — each worker's BLAS is no longer
multi-threaded, so 8 workers on 32 logical cores are no longer oversubscribed,
and the residual is now the pool's own startup and serialization rather than
thread contention. Note also that the serial baseline got *faster* under the
pin (121.75 s → 115.80 s): multi-threaded BLAS was pure overhead at these
problem sizes, on the serial path as well as the pooled one.

### 7.4 The gate the brief asked for

`tests/test_determinism.py::TestTwoThreadCountsAreNeverOneAnswer` runs one
GARCH-t cell in **two subprocesses at two real thread counts** — a subprocess
because OpenBLAS reads its count when the library loads, so an in-process
`monkeypatch` would change the recorded pin without changing a single
arithmetic operation, i.e. would make the test pass while testing nothing — and
asserts identical numbers **or** different hashes. Verified non-vacuous on this
box: 40 origins, numbers genuinely differ (2.4e-5 relative), hashes differ, so
the disjunction is carried by the hash clause. Before D-032 that same
comparison was "same hash, different numbers".

### 7.5 Leakage audit — `main...feat/p3-determinism`

Run because the diff touches **caching** (item 9) and **evaluation** (the row
schema); it touches no loader, splitter, feature or scaler at all.

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Index arithmetic at boundaries | PASS | No index arithmetic added. `_run_block`'s loop over `task.origins` and its `for horizon, target_index in enumerate(origin.test, start=1)` are untouched; the diff adds one dict key to the row built inside it. |
| 2 | Splitter monopoly | PASS | No new train/test indices anywhere. `GARCH._model(arr)` receives the array `_run_block` already passed to `fit`; `_count_fits` reads a stored frame and never indexes a series. |
| 3 | Feature lags | PASS | No features added or changed. |
| 4 | Transforms and scalers | PASS | `nu_bounds` is a box constraint on the optimizer, `fit_tol` its stopping rule — both per-fit, both inside the window `fit` was handed. `_BoundedStudentsT.bounds()` ignores its `resids` argument entirely and returns a constant. |
| 5 | Target construction | PASS | Untouched. |
| 6 | Refit schedule | PASS | Unchanged. `fit_status` is *derived from* the schedule (a property of the fit at `fit_origin`, carried across the block by `update`, which runs no optimizer) and never feeds back into it. |
| 7 | TSFM context windows | PASS | Untouched. |
| 8 | Calendar alignment | PASS | Untouched. |
| 9 | **Caching** | PASS | The focus item, and the change makes the key **strictly more** discriminating: `build_config` gains `environment`, never loses a field, so any two configs that hash equally now hashed equally before. A key that splits more finely cannot serve an artefact it previously would not have. Verified directly rather than argued: one cell run three times into one store — `threads=1` (computed), `threads=1` again (**cached**, same hash), `threads=4` (**not cached**, new hash, store now holds two fragments). |
| 10 | Survivorship & selection | PASS | No series selection changed. Worth stating for `fit_status`: it is *recorded*, never acted on. Nothing drops, reweights or reruns a cell because of its fallback rate — doing so would be selection on an outcome, and `docs/design.md` carries that as an open protocol question precisely so it is decided deliberately rather than in code. |

**Canary.** `tests/test_m1_smoke.py::test_future_corruption_cannot_change_past_forecasts`
(corrupt everything strictly after T; every forecast for targets ≤ T must be
bit-identical) and its inert-proof companion `test_the_canary_can_actually_fail`
both pass on this branch.

**Verdict: PASS, no findings.** One residual note, not a leak: `fit_status`
carries the fitted `nu` in its detail string, so a results table now exposes an
estimated parameter it did not before. That parameter is estimated from the
window ending at `fit_origin` and is written after the fit, so it moves no
information backwards; it is an observation of the run, and the tests pin that
it never reaches `config_hash` and that a diagnostic which raises cannot turn a
scored origin into a missing one.

---

## 8. Drift flagged, not edited

- `docs/P3_RUNNER.md` §7 ("Found while measuring — the OpenBLAS thread count
  changes GARCH's numbers") is now superseded in two ways: its recommendation
  is taken (D-032), and its mechanism sentence — "threaded BLAS reorders a
  reduction by an ulp, and `arch`'s SLSQP amplifies it into a different local
  optimum" — is right as far as it goes but does not name *why* the Student-t
  is 4 orders of magnitude more sensitive than the normal. §2 above supplies
  that. `docs/P3_RUNNER.md` is another branch's evidence record and is not
  edited.
- `docs/P3_RUNNER.md` §4's "Single-threading the workers gets to 4.7×–5.1× —
  and is *rejected*, because it changes GARCH's numbers" is superseded: the
  numbers were allowed to change, deliberately and under a version bump, and
  §7.3 above measures 4.80×/5.08×, inside that predicted range.
- `docs/research_design.md` H4 says "≥5× on 8 cores"; §7.3 measures 4.80× and
  says so. The design doc is a mirror and is not edited.
- `docs/M1_REPORT.md` §4.3's account of GARCH re-conditioning is unaffected;
  `update` still runs no optimizer, and `tests/test_models_update.py` still
  pins that `fix` reproduces the fitted forecast to the bit.
