# Phase-2 core integration report — volbench v0.3.0-p2core

Branch `m2/p2-integration`, 2026-08-25. Four streams merged into the M2 tree
(`main` at v0.2.0, `923d3c7`), one `--no-ff` merge each, in this order:

| # | Branch | Commits | What it built |
|---|---|---|---|
| 1 | `feat/p2-models-classical` | 3 | AutoETS / AutoARIMA (statsforecast) and LightGBM on log realized variance; the shared `_rv` retransformation |
| 2 | `feat/p2-models-tsfm` | 9 | Chronos, TimesFM, Moirai, TimeGPT zero-shot adapters on one contract; the PatchTST baseline; the `tsfm` / `torch-cpu` extras |
| 3 | `feat/p2-inference` | 2 | Diebold-Mariano (HLN), the Model Confidence Set; Kupiec, Christoffersen, FZ0, expected shortfall |
| 4 | `feat/p2-data-panel` | 4 | The D-004/D-012 panel, crisis tags, diagnostics, the panel report; bulk-archive Stooq parsing |

All four branched from the same commit, so every conflict was cross-stream.
Real conflicts (expected, resolved by union): `.github/workflows/ci.yml`,
`Makefile`, `pyproject.toml` (git had produced two
`[project.optional-dependencies]` tables — invalid TOML), `uv.lock` (taken
from the tsfm side, then regenerated once after all four merges: 8 packages
added for `classical`, nothing else moved) and the docstring of
`src/volbench/models/__init__.py`. Streams 3 and 4 merged clean.

This report follows the M1 pattern: what exists, where each stream deviated
from its brief, what the streams disagreed about, what integration changed,
what was measured, and what was deliberately **not** decided. Nothing is
papered over; the items the brief required recording are §3.

## 1. What exists

Package: 39 source modules, 11.1k lines (`src/volbench`), 960 tests (41
files); `mypy --strict` over `src` and `tests/test_model_interface.py`.
Optional backends in three extras (`classical`, `tsfm`, `torch-cpu`; §4).

### Stream A — classical log-RV models (`models/sf.py`, `models/lgbm.py`, `models/_rv.py`; 137 tests)

- `AutoETSRV` (statsforecast `AutoETS`, `model="AZN"`, `season_length=1`) and
  `AutoARIMARV` (Hyndman-Khandakar stepwise, `aicc`, `kpss`, no
  approximation; every search setting in `spec()`), both on `log RV`.
- `LightGBMRV`: L2 boosting of `log RV_{t+1}` on lags 1..22 of log RV plus
  HAR's weekly/monthly aggregates (24 features; a HAR-equivalent linear
  function exists inside the feature set). No scaler, no early stopping, no
  validation split — each a documented leak — and their absence is asserted
  (a window's design matrix is bit-identical whether or not the array
  continues afterwards). `deterministic=True`, `force_row_wise=True`,
  `num_threads=1`, one seed; byte-identical serialized boosters pinned.
- `SupportsUpdate` implemented *exactly* for all three: statsforecast's
  `forward` re-filters at fixed parameters; LightGBM's `update` moves the RV
  buffer under fixed trees. Verified behaviourally (update on the fit window
  reproduces the fit; a shifted window moves it; the fitted object is not
  mutated).
- `_rv.py`: one implementation of the log-to-variance retransformation —
  Duan (1983) smearing (default) and the Gaussian `exp(σ²/2)` arm, both
  config-hashed and in the model `name`. §3.2 records its known optimism.

### Stream B — foundation models and PatchTST (`models/tsfm_*.py`, `models/patchtst.py`, `models/_patchtst_net.py`; 129 tests, 13 of them opt-in)

- One contract (`tsfm_common.py`): `fit` records the trailing
  `context_length` of a realized-variance series (zero-shot, D-005 — nothing
  is estimated); `predict(h)` takes the **mean of the checkpoint's own RV
  quantile grid** at `t+h` as the variance forecast and emits `Normal(0,
  sqrt(vhat))`; `update` is exact context extension, so `refit_every` changes
  no number (pinned byte for byte at 1 and 21). Crossing is rearranged and
  negative quantiles clipped, both counted in the fitted spec. `spec()`
  carries the checkpoint id, the resolved commit hash of the cached weights
  (read from the HF cache, no network), dtype and package versions, so the
  config hash moves with the weights; `device` is not hashed.
- `Chronos` (Bolt-small default, Chronos-2 by checkpoint; 9 or 21 direct
  quantiles), `TimesFM` (2.5-200M; point head split off and recorded, never
  scored; README-recommended forecast options, all hashed), `Moirai`
  (2.0-R-small via the array-level `Moirai2Forecast` predict, never the
  gluonts path), `TimeGPT` (triple-gated: `enabled=True`, `NIXTLA_API_KEY`,
  `@pytest.mark.timegpt`; cannot pin remote weights and says so in `spec()`).
- `PatchTST`: small fixed channel-independent PatchTST (~20k parameters) on
  instance-normalized log RV; bounded, hashed training budget with early
  stopping on the chronologically-last 20% of windows; Duan smearing per
  horizon; **no `update`** (frozen between refits, the documented exception
  to the update invariant). Deterministic by construction; §3.4 records the
  per-device-class caveat.
- `tests/conftest.py`: `tsfm`, `timegpt`, `gpu` markers, skipped by default
  and unconditionally under `CI`; `tests/tsfm_fakes.py` gives every adapter a
  weight-free contract test in the default suite.

### Stream C — comparison inference and VaR backtests (`inference.py`, `backtests.py`; 81 tests)

- `diebold_mariano`: rectangular or Bartlett window truncated at `h-1`,
  Harvey-Leybourne-Newbold factor, `t_{n-1}`; DM's own rule for a
  non-positive variance estimate (flagged on the result); at `h=1` exactly
  the one-sample t (pinned); size by simulation within 3.5 MC standard errors.
- `model_confidence_set`: Hansen-Lunde-Nason sequential elimination, `T_R`
  (default) or `T_max`, moving-block bootstrap by hand with **no
  wrap-around** (HLN's appendix uses a circular variant — documented),
  Politis-White automatic block length mirrored from and pinned against
  `arch.bootstrap.optimal_block_length`, `B=10 000`, `alpha=0.10`.
- `loss_matrix` / `dm_matrix` / `compare_models`: the MCS is the primary
  "who wins" tool and is returned together with the pairwise DM matrix,
  whose p-values are not multiplicity-corrected. Missing rows drop
  pairwise-complete (DM) / listwise (MCS), `n_dropped` recorded; cells from a
  store must share the same series bytes before their origins are aligned.
- `kupiec_pof`, `christoffersen` (`LR_cc = LR_uc + LR_ind` exactly; a NaN hit
  removes its neighbouring transitions rather than splicing across the gap),
  `fz0_loss` (Patton-Ziegel-Chen 2019 eq. 6, pinned against their figures
  and the `L(kY,kv,ke) = L + log k` identity), `expected_shortfall`,
  `var_backtest`; `SmallSampleWarning` below 10 expected exceedances.
- Touches nothing on the scored path: `make reproduce` was byte-identical to
  `main` on the branch.

### Stream D — the evaluation panel (`data/panel.py`, `crisis.py`, `diagnostics.py`, `build_panel.py`, Stooq bulk parsing; 148 tests)

- `EQUITY_PANEL` / `CRYPTO_PANEL`: seven indices Stooq still serves (NDX,
  DAX, CAC, NKX, HSI, TWSE, KOSPI), the D-012 ETF stand-ins SPY/DIA/ISF for
  the SPX/DJI/FTSE-100 slots, BTC/ETH from Binance minute bars. Stooq is
  never fetched programmatically: the equity arm reads hand-downloaded bulk
  archives under a `raw_root` outside the repo; `tests/test_licensing_guard.py`
  asks git that the data trees stay ignored and untracked.
- Targets built on each file's full history and trimmed to the window
  afterwards (the first in-window day keeps a genuine previous close).
- Bar quality: sub-1e-5 relative violations clamped and counted; larger ones
  (a close outside its own session — two feeds disagreeing) left as they are,
  flagged, range targets set NaN, **row kept** so `run_backtest` records a
  `missing_reason`.
- `crisis.py`: dates only; the module's AST is checked for imports of the
  forecasting stack; the 2025-26 window is undated (D-004: fixed at freeze).
- `docs/PANEL_REPORT.md`, regenerated by `build_panel.py`, no hand-entered
  figures. Its findings are the pending decisions in §8.
- The leakage audit on that branch found and fixed two things:
  `crisis_coverage` re-derived the splitter's arithmetic and was off by one
  (now takes the union of the splitter's own `test` indices), and
  `repair_bars` read classification arrays through a `to_numpy()` view that
  clamping then rewrote (found on the real NKX 2020-10-01 bar).

### Added at integration (this branch)

- Public surface: every adapter re-exported from `volbench.models` and
  `volbench`; optional backends imported lazily inside `fit`
  (`tests/test_optional_backends.py` runs `import volbench` with all seven
  backend packages blocked). Root also exports the inference and backtest
  entry points and the two proxy pieces stream D made public.
- `tests/test_model_interface.py`: `AutoETSRV`, `AutoARIMARV`, `LightGBMRV`
  in the strictly-typed conformance list (72 tests over 12 models; the
  same-object test — `volbench.evaluate.ForecastModel is
  volbench.models.base.ForecastModel` — holds).
- PatchTST retransforms through `_rv` (§6.3).
- `STOOQ_INDEX_SYMBOLS` without the CFD entries (§6.4).
- `benchmarks.toy`: +AutoETS, +AutoARIMA, +LightGBM (8 models);
  `benchmarks.smoke_tsfm` / `make smoke-tsfm` for the heavy four (§7).
- `.github/workflows/ci.yml`: push trigger on `main`, `feat/**`, `m2/**`;
  `--extra classical --extra torch-cpu` on every leg.
- One `[tool.uv]` block (§4); `EXTRAS` variable in the Makefile.
- Version 0.2.0 → 0.3.0; eight toy identities re-pinned (§7).
- `docs/decisions.md` D-021..D-026 (renumbered from the provisional D-017..D-022
  on `feat/p2-protocol`; see that file's numbering note); `docs/design.md`
  reconciled in one pass — all four streams had flagged its drift.

## 2. Integration mechanics — what was reconciled and how

| Conflict | Classical said | TSFM said | Resolution |
|---|---|---|---|
| CI extras | `--extra classical` on every leg | `--extra torch-cpu` on every leg | both, on every leg; `tsfm` never |
| Makefile | `--extra classical` on every `uv run` | *no* extra: `uv run --extra` syncs to exactly the named extras and would swap the GPU box's cu121 torch out | `EXTRAS ?= --extra classical`, `UV_RUN := uv run $(EXTRAS)`; the GPU box sets `EXTRAS="--extra classical --extra tsfm"` |
| `[tool.uv]` | `override-dependencies = ["pandas>=3.0.5"]` | numpy/scipy/pandas overrides + `conflicts` + `dependency-metadata` + `sources` + two indexes | one commented block, §4 |
| `models/__init__` | do **not** re-export sf/lgbm (eager backend imports) | re-export (lazy imports) | re-export everything, lazy everywhere (D-022) |
| `uv.lock` | classical additions | tsfm additions (2 815 lines) | tsfm side taken at the merge, regenerated once at the end |

The merge commits carry the union resolution only; every deliberate change
(lazy imports, exports, `_rv` unification, CFD retirement, benchmarks,
version, docs) is its own conventional commit on top, so the history reads
as merges + decisions.

## 3. Records the brief required

### 3.1 TSFM grid-mean truncation — a known, bounded downward bias

`predict(h)` for the four zero-shot adapters uses the mean of the
checkpoint's quantile grid with flat tails outside the outer levels — the
same estimator `forecast_moments` applies to a `QuantileGrid` (pinned equal
bit for bit). That estimator is the one that understated a Student-t
GARCH's variance by ~8% at ν=5 and ~24% at ν=3 in M1 (report §4.2), and
D-014 fixed *that* case by emitting a parametric `StudentT` instead. Here
no parametric object exists: the grid is what the checkpoint returns
(Chronos-Bolt/Moirai 9 levels 0.1..0.9, Chronos-2 21 levels, TimesFM
0.1..0.9). The bias is **downward**, monotone in the tail mass outside the
outer quantiles, and identical in family to the D-014 bug. On the toy
series the TSFM QLIKE values (0.176–0.178) sit between EWMA (0.162) and
HAR (0.182), so nothing looks odd yet; **revisit if TSFM QLIKE looks odd
on the real panel**, especially in crisis windows where the outer quantiles
carry more mass. Tail-extrapolating the grid would be the same "patch on a
lossy representation" D-014 rejected. Tracked in `docs/design.md` open
questions.

### 3.2 LightGBM's in-sample smearing optimism — 0.28 vs 0.38; out-of-fold factors are open

Duan's factor wants residuals that behave like draws from the forecast-error
distribution. An ensemble shrinks its own residuals, and a shrunken residual
set drives the factor to 1, quietly turning the variance forecast back into a
median forecast. Measured on the toy fixture at 500-observation windows:
LightGBM's stock shape (300 rounds, 31/15 leaves, `min_data_in_leaf=20`)
puts the in-sample log-space residual variance at **0.015 against a realized
one-step forecast-error variance of 0.42** — factor 1.008 where HAR's is
1.207. The shipped defaults are a deliberately small ensemble (100 rounds,
depth-2 trees, `min_data_in_leaf=60`, `lambda_l2=5`) that lands at **0.28
vs 0.38**, an optimism of HAR's own order; `tests/test_models_lgbm.py::
TestRetransformation::test_the_ensemble_does_not_memorize_its_own_residuals`
fails if the capacity is raised. The understatement is bounded, not
eliminated. **Open item: an out-of-fold factor** — estimated on a *temporal*
fold inside the training window, never a random one — is the honest fix and
a Phase-2 modelling decision (D-025 revisit-if). Note the same critique
applies in principle to PatchTST's training-residual factor; its early
stopping on a chronological hold-out is a partial mitigation only.

### 3.3 Moirai's `input_scale = 1e4` is a necessity, not a convenience

Moirai-2's scaler computes `sqrt(var + 1e-5)`; at raw daily-variance units
(~1e-4) the epsilon dominates and a whole context flattens into a constant.
Measured on the GPU (tsfm-marked test): at raw units the q10–q90 spread
collapses below 5% of the level; at 1e4 it exceeds 50% and is stable from
there upward (1e4 vs 1e6 agree). Chronos and TimesFM are indifferent. The
scale is applied to all four so the contract has one unit and is recorded
in `spec()`; it is a fixed convention, not a tuned hyperparameter.

### 3.4 PatchTST reproduces per device class, and `device` is not hashed

Bit-identical twice on CPU (CI smoke) and twice on the 4090 (gpu-marked),
but dropout draws from each device's own RNG stream, so a CPU fit and a
CUDA fit of the same window are different realizations (~1e-2 in the
forecast); with `dropout=0` they agree to ~1e-8. Since `device` is
deliberately outside `spec()`, two `ResultsStore` fragments with **one config
hash can legitimately differ** if computed on different device classes. For
the paper this means every PatchTST number states its device class, and a
grid must not mix them under one hash. Whether to hash the device class
instead is a protocol call (design.md open questions), not taken here.

### 3.5 The `pandas>=3.0.5` override and its evidence

statsforecast (through 2.1.1) declares a preemptive `pandas<3.0.0`; uni2ts
2.0.0 and nixtla declare `pandas<3` likewise. volbench's core requires
`pandas>=3.0.5`, so without the override the `classical` and `tsfm` extras
do not resolve. The cap is a guess, not an observed break, and the evidence
is a test, not a hope: `tests/test_models_sf.py::TestBackendCompatibility::
test_both_backends_run_their_full_path_under_the_installed_pandas` runs the
two statsforecast adapters' entire fit/forward/predict path under the
installed pandas (3.0.5 in the lock) and passes on all three interpreters;
the tsfm-marked suite does the same for uni2ts/nixtla on the GPU box. If a
pandas release ever breaks a backend, it breaks there first and the override
gets revisited instead of silently shipping. The override applies to every
requirer at resolution time — that is its point and its risk.

### 3.6 Found by CI at integration — byte-identity holds within a numpy SIMD kernel family

Not in the brief; found because the widened CI trigger ran the suite on the
branch. The pinned-identity test (`tests/test_recondition.py`) failed on the
3.13 leg of the first CI run and on the 3.12 leg of the second, and passed
on the 3.11 leg of the first — same commit, same lock, same fixture. The
config dump added for the second run shows exactly one difference between an
offending runner and this box: the **proxy's content digest**
(`overnight_plus_range`, `0d312ed0…` here vs `5758b0a3…` there). The
returns digest (also `np.log`), the fit-series digest, the spec, the splitter
and the version are identical.

Cause: numpy 2.x dispatches float64 `log`/`exp` to AVX-512-only SIMD kernels
on machines that have them, and those are not bit-identical to the x86-v3
(AVX2/FMA) kernels in the last ulp for some inputs. GitHub's `ubuntu-latest`
pool mixes CPU generations, so which kernel a job gets is a lottery; the
five logs per day in the range estimator hit a differing input where the 700
return logs did not. This box (i9-13900KF, no AVX-512) computes with the v3
kernels, and disabling them locally (`NPY_DISABLE_CPU_FEATURES=X86_V3`,
falling back to the v2 baseline) leaves every digest unchanged — the v2/v3
kernels agree on this data; the v4 ones do not. **M2's green CI on `main`
was therefore runner luck**, not evidence of cross-ISA bit-identity.

What this means and what was done:

- Content digests of *computed* proxies (D-016's estimator is a computation
  over the OHLC bars) are ISA-dependent at the last bit, so the byte-identity
  `make reproduce` claims is a claim **within one numpy kernel family**, not
  across every CPU. That is a property of IEEE arithmetic plus SIMD
  dispatch, not of volbench, and it is now stated in `docs/design.md`
  (invariant 3) rather than implied away.
- CI and the Makefile's `reproduce` export
  `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`, which pins an
  AVX-512 machine to the v3 kernels the committed identities were computed
  with; on a machine without those features the setting is a silent no-op
  (verified). The pinned identities are unchanged.
- The cache-identity control is unaffected in the direction that matters:
  a digest mismatch across machines makes the store *miss* (recompute), never
  serve the wrong artefact. The cost is a false miss, not a false hit.
- Scores are affected at the ~1e-16 level only; no ranking or reported
  number in this report changes. The paper's grid should nonetheless be run
  on one kernel family and say so (D-026).

## 4. The three uv mechanisms, reconciled

Two streams introduced three resolver mechanisms independently. They now
live in one commented `[tool.uv]` block in `pyproject.toml`, with the order
of preference written down:

1. **`override-dependencies`** — for a stale *upper bound* a backend declares
   against a package volbench pins a floor for: `numpy>=2.0`, `scipy>=1.17.1`
   (uni2ts and its gluonts/datasets pins declare `numpy<2`, `scipy~=1.11`),
   `pandas>=3.0.5` (§3.5). Each has a runtime compatibility test.
2. **`dependency-metadata`** — for a bound that *cannot* be overridden because
   the same package is also pinned per-extra to an index: uni2ts's
   `torch<2.5`. An override replaces every torch requirement, the extras'
   own index-scoped `torch==2.5.1` included, and is unaware of the extra
   conflict, so the resolver either lost the PyTorch index or saw both (three
   variants tried on the tsfm branch and recorded). Re-declaring uni2ts's
   published requirement list with the cap removed lifts it where it lives.
3. **`conflicts` + `sources` + two explicit indexes** — for two builds of one
   package that cannot coexist: `tsfm` carries `torch==2.5.1` from the cu121
   index (the build the 4090's driver 535 can load; PyPI's current torch
   needs driver ≥ 580), `torch-cpu` the same version from the CPU index, and
   the two extras are declared to conflict.

Dropped: nothing. Duplicated: nothing (the classical stream's pandas
override is the same entry the tsfm stream had). `uv lock` after all four
merges added exactly the 8 packages the `classical` extra needs.

## 5. Where each stream deviated from its brief

### Classical
- **Kept its adapters out of `volbench.models`** (eager backend imports)
  rather than exporting them — reversed at integration (D-022, §6.1).
- **`SupportsUpdate` implemented** where the brief made it conditional on the
  backend re-filtering at fixed parameters; it can, and one sharp edge was
  handled rather than absorbed: `forward_ets` re-estimates the innovation
  variance, so the Gaussian arm reads the h-step variance from the scheduled
  fit's own `predict`, never from `forward`.
- **Defaults are not LightGBM's stock shape** — measured, §3.2.
- The leakage audit added a guard: `_feature_row` rejects `t < 21` (numpy
  reads a negative slice start as `len + start`, a silently wrong-length row
  — unreachable through the public API, exactly the shape of a leak).
- Recorded a modelling property: ETS/ARIMA `forward` re-runs the filter from
  the *initial* state of the last fit rather than carrying state forward
  (what R does; immaterial at 500-observation windows, not at short ones
  with a near-zero smoothing parameter).

### TSFM / PatchTST
- **RV-fed, not return-fed.** The research design lists these as adapters
  of a generic kind; as built they take a realized-variance series like HAR.
  The interface is uniform in type and not in meaning, now for eight of the
  twelve models (design.md, M1 risk 2).
- **PatchTST over N-BEATS**, on the flagged assumption that architectural
  proximity to the patch-based TSFMs is the more useful comparison.
- **PatchTST has no `update`** — the one model that runs frozen between
  refits, recorded as the documented exception to the update invariant.
- **TimeGPT stays out of the headline** (cannot pin weights; paid API).
- The `torch<2.5` cap moved from an override to `dependency-metadata` after
  the override broke the index selection (§4).
- Re-implemented `_rv`'s validation and retransformation locally (four
  lines) and flagged it — unified at integration (§6.3).

### Inference
- **No deviation in scope**; two documented choices: bootstrap blocks are
  contiguous forward-running windows with no wrap-around (HLN's appendix is
  circular), and the FZ0 loss enforces PZC's domain (`ES < 0`, `ES ≤ VaR`)
  rather than returning NaN silently.
- Consumes rows only; `make reproduce` unchanged on the branch.

### Data panel
- **D-012's ETF substitution is applied** (SPY/DIA/ISF) although D-012 is
  not mirrored in `docs/decisions.md` here — the planning machine holds it.
- **`docs/design.md` drift flagged, not edited**, per the mirror rule;
  reconciled in this branch under the CLAUDE.md carve-out.
- **Findings recorded, not acted on** (§8): ISF starts 2015 (D-012's
  fallback trigger fired); the GFC arm is largely inside the warm-up; the
  overnight share is 33–51%, not ~9–15% (that figure came from the toy
  generator, not from market data — "D-016 was more right than its own
  rationale claimed"); 14 exactly-zero primary targets (HSI's stale opens on
  monotone bars — a log cannot take them); TWSE's 1.5% inconsistent bars;
  crypto's target is 5-minute RV, whose overnight term is 0.01% of the total.
- Made `rogers_satchell` and `overnight_variance` public so the report's
  decomposition is literally the target's two summands.
- Survivorship flagged at the design level (the ten equity series are
  instruments liquid in 2026, chosen in 2026) — belongs in the paper's data
  section, not fixed in code.

## 6. Cross-stream disagreements surfaced at integration

### 6.1 Export policy — RESOLVED (D-022)
Classical: "importing them from this package root would make `import
volbench` fail for anyone who installed the core library". TSFM: lazy
imports, re-export. Both were right about their own code; the brief's
"exports for every new adapter" plus lazy imports satisfies both concerns.
Cost: `sf.py`/`lgbm.py` import their backends inside `fit` and reference the
backend types under `TYPE_CHECKING` only; mypy still checks the real
statsforecast/lightgbm signatures because the `classical` extra is installed
wherever mypy runs. Pinned by `tests/test_optional_backends.py`.

### 6.2 Makefile extras — RESOLVED
Classical put `--extra classical` on every `uv run`; TSFM explained why any
extra there would silently swap the GPU box's torch. The `EXTRAS` variable
is the reconciliation; `make smoke-tsfm` names its extras explicitly because
they are not negotiable.

### 6.3 Two copies of the RV plumbing — RESOLVED (D-025)
PatchTST re-implemented `validated_rv` and `exp(mu)·factor` locally and
computed its per-horizon Duan factor in torch. It now uses `_rv`'s
`validated_rv`, `variance_from_log` and `smearing_factor` (per horizon
column, in numpy). The smoke run's PatchTST numbers are unchanged at printed
precision before and after (CRPS 0.005856, QLIKE 0.182143), and its
determinism tests still pass on CPU and GPU; `_rv.smearing_factor` drops
non-finite residuals rather than propagating them, which PatchTST's copy did
not — a strict improvement for a residual set that should never contain one.

### 6.4 Two asset lists — RESOLVED (D-024)
`STOOQ_INDEX_SYMBOLS` (M1, with SPX/DJI/FTSE → CFD proxies) vs
`EQUITY_PANEL` (D-012, with SPY/DIA/ISF). The CFD entries are gone; a test
pins every remaining entry against the panel's ticker. `docs/data_licenses.md`
still describes the CFD situation as "flagged for a human" — it is a
read-only mirror here, and the planning machine should update it to record
that D-012 answered the question.

### 6.5 Raise vs fall back on a bad origin — OPEN, unchanged
GARCH falls back to EWMA; HAR raises; every Phase-2 model raises (the
evaluator records `fit_error@origin`). The panel's zero-variance targets
will hit exactly this seam on HSI (§8.1). Left as at M1.

### 6.6 What "the same model" means across devices — OPEN
§3.4. Not a disagreement between streams but between PatchTST and the
content-addressed store's promise that one hash means one artefact.

### 6.7 HAR is now the odd one out — OPEN
Every Phase-2 log-RV model retransforms through `_rv` with smearing by
default; HAR still uses its own Gaussian `resid_var`. Moving it changes its
numbers and hash, so it is a modelling decision, not an integration side
effect (design.md open questions).

## 7. Measured — the toy benchmark and the smoke run

Synthetic series (`make_toy_asset`), 200 rolling origins, window 500, h=1,
scored against `overnight_plus_range` (D-016). **No number here belongs in
the paper**; these are wiring and plausibility checks.

### 7.1 `make reproduce` (8 models, 3.11, ~32 s wall for the benchmark step)

| label | model | CRPS | log score | QLIKE | mean σ̂ | hit 1% / 2.5% / 5% |
|---|---|---|---|---|---|---|
| ewma | ewma | 0.005820 | −3.1859 | 0.1616 | 0.01078 | .005 / .020 / .040 |
| lgbm | lightgbm_rv-smearing | 0.005838 | −3.1726 | 0.1757 | 0.01122 | .005 / .015 / .045 |
| autoets | autoets_rv-smearing | 0.005847 | −3.1686 | 0.1818 | 0.01131 | .005 / .015 / .040 |
| autoarima | autoarima_rv-smearing | 0.005849 | −3.1674 | 0.1814 | 0.01129 | .005 / .015 / .040 |
| har | har_rv | 0.005860 | −3.1599 | 0.1823 | 0.01154 | .005 / .020 / .035 |
| garch11_t | garch(1,1)-studentst | 0.005872 | −3.1596 | 0.1729 | 0.01188 | .005 / .020 / .035 |
| garch11 | garch(1,1)-normal | 0.005873 | −3.1597 | 0.1729 | 0.01187 | .005 / .020 / .035 |
| naive | naive_rw_vol | 0.006013 | −3.0784 | 0.3028 | 0.01354 | .010 / .025 / .030 |

The five M2 models' scores are unchanged to every printed digit from the
0.2.0 run — only their hashes moved, with the version. The three classical
models land where they should on a series whose true dynamics are HAR-like:
between EWMA and HAR, with LightGBM's smaller mean σ̂ consistent with §3.2's
residual optimism (a factor closer to 1 than HAR's). Two consecutive
`make reproduce` runs were byte-identical on all eight fragments (§9).
Pinned identities (`tests/test_recondition.py`):

```
autoarima e26f67ea…a6a7ecd   autoets 74904119…b35e7e2f   ewma a3b64eef…e8542214
garch11   c8a59b3a…f8def3a9   garch11_t 0fd9610c…662e816a7 har 96d1222c…aa8c793f58
lgbm      57e89612…3eeb8da0   naive   7b41390a…0fa6283cd7
```

### 7.2 `make smoke-tsfm` (4 models, 4090, cu121 torch 2.5.1, ~25 s wall, refit_every=21)

| label | model | CRPS | log score | QLIKE | mean σ̂ | hit 1% / 2.5% / 5% |
|---|---|---|---|---|---|---|
| timesfm | timesfm_2_5_200m_pytorch | 0.005841 | −3.1742 | 0.1757 | 0.01094 | .010 / .020 / .040 |
| chronos | chronos_bolt_small | 0.005852 | −3.1670 | 0.1765 | 0.01113 | .010 / .025 / .045 |
| moirai | moirai_2_0_r_small | 0.005853 | −3.1669 | 0.1779 | 0.01110 | .005 / .020 / .040 |
| patchtst | patchtst | 0.005856 | −3.1660 | 0.1821 | 0.01123 | .010 / .025 / .045 |

Config hashes at 0.3.0: chronos `c3654673…3f84dee`, moirai `499140b6…cdbe0a1`,
patchtst `5073d3b4…c24bf26`, timesfm `7356edfd…7aa09`. Two consecutive runs
byte-identical on all four fragments, at 0.2.0 and again at 0.3.0. All 800
rows scored (no `missing_reason`). The zero-shot models on a synthetic
HAR-like series score like a well-tuned EWMA/HAR — a plausibility check on
the contexts and the unit convention, nothing more; TSFM QLIKE is not odd
here (§3.1's trigger is the real panel).

## 8. Deliberately NOT decided here — pending protocol decisions

The brief excludes these from the integration; each ships in a dedicated
follow-up against the merged tree, with its own D-entry. Listed so they are
not lost.

> **Items 1-3 are now closed**, on `feat/p2-protocol` at v0.4.0: D-018
> (invalid-target policy), D-019 (a 500-observation fit window, 1000 as the
> robustness arm) and D-020 (the FTSE 100 leg dropped; the panel is 11
> assets). §11 records what shipped and what it cost. Items 4 and 5 are
> unchanged and still open.

1. **Invalid-target policy.** 14 exactly-zero primary targets (HSI's stale
   opens meeting monotone bars: Rogers-Satchell is exactly zero on a
   monotone bar, and the overnight term is zero when the open equals the
   previous close) and TWSE's inconsistent bars. Options recorded by the
   panel stream: let HAR fail and report NaN rows; NaN the zero days like
   any other bad bar (its recommendation); floor the target (changes the
   estimator); drop HSI. Every log-RV model (HAR, AutoETS, AutoARIMA,
   LightGBM, PatchTST, the TSFMs' `log` is not involved but Moirai's scaler
   is) will hit the same seam.
2. **Rolling-window length.** Under 500 observations the GFC arm is mostly
   warm-up: 140–149 GFC days per equity series in the panel, 31–86 scored,
   none for ISF/BTC/ETH. COVID, 2022 and Aug-2024 are fully scored.
3. **The FTSE-100 slot.** ISF starts 2015-03-04 — 52% of the longest equity
   series — so D-012's fallback trigger has fired; keep, re-source, or drop
   from H3.
4. Also from the panel report, smaller: the 1e-5 bar-repair threshold as a
   decision entry; the Aug-2024 window's exact dates; the 2025-26 window
   (D-004: at freeze); correcting the ~9–15% overnight-share expectation in
   the planning documents; NKX's overnight/intraday correlation.
5. From the model streams: out-of-fold smearing factors (§3.2); HAR onto
   `_rv` (§6.7); hashing the device class (§3.4); tail treatment of the
   TSFM grid (§3.1); per-model refit-schedule overrides (carried from M2).

## 9. Gates

Filled in from the runs on this branch before merge; see the final section
of this file's git history for the CI run on the merge commit.

- `ruff check .` — clean on 3.11 / 3.12 / 3.13.
- `mypy --strict` (`src` + `tests/test_model_interface.py`) — clean on
  3.11 / 3.12 / 3.13 (40 source files).
- Full CPU suite: see §9.1.
- `make reproduce` at 0.3.0: green; the eight fragments byte-identical
  between two from-scratch runs (§9.1).
- tsfm / gpu opt-in suites on the 4090: §9.1.
- Leakage check over the full integration diff: §9.2.
- CI on the merge commit: §9.3.

### 9.1 Local runs

All on this box (RTX 4090, driver 535, torch 2.5.1+cu121 in the 3.11 venv;
3.12/3.13 venvs built with the CI extras), on the tree at the commit before
this docs commit.

| Leg | Extras | ruff | mypy --strict | pytest |
|---|---|---|---|---|
| 3.11 (`make reproduce EXTRAS="--extra classical --extra tsfm"`) | classical + tsfm | clean | clean, 40 files | **931 passed, 29 skipped** (the opt-in `tsfm`/`gpu`/`timegpt` tests and the TimeGPT-key tests), 7 m 39 s |
| 3.12 (`CI=true`, `--extra classical --extra torch-cpu`) | classical + torch-cpu | clean | clean | all passed, exit 0 |
| 3.13 (`CI=true`, `--extra classical --extra torch-cpu`) | classical + torch-cpu | clean | clean | all passed, exit 0 |
| 3.11 opt-in (`VOLBENCH_RUN_TSFM=1 VOLBENCH_RUN_GPU=1 pytest -m "tsfm or gpu or timegpt"`) | classical + tsfm | — | — | **28 passed, 1 skipped** (TimeGPT: no `NIXTLA_API_KEY` set, by design) |

`make reproduce` rebuilt `data/toy_benchmark/` from scratch (fixture
regenerated and diffed clean against the committed CSV, results directory
deleted first) and the eight parquet fragments are **byte-identical** to the
run that produced the pinned identities in §7.1. `make smoke-tsfm` twice at
0.3.0: four fragments byte-identical (§7.2).

The first pass of these gates found two test failures, both fixed in
`f5bd993` before the pass above: `test_toy_targets` hard-coded HAR as the only
variance-fed model (it now pins the set by label), and `test_smoke_tsfm`
called `spec()` on real-backend adapters, which reads the backend's version
and so needs the `tsfm` extra CI never installs (found on the 3.12/3.13 legs
— exactly the gap the widened CI trigger closes). That first pass also
illustrates why "byte-identical" must be checked against a from-scratch
rebuild: with `check` failing, `reproduce` never reached the benchmark step
and a naive digest comparison would have compared run 1 with itself.


### 9.2 Leakage check (`.claude/skills/leakage-check`, full diff `main...m2/p2-integration`)

Run over `git diff main...m2/p2-integration` (52 files) with the skill's ten
items; each stream had already run the audit on its own branch and the two
findings below marked FIXED are theirs. The integration commits themselves
add no code path that reads past an origin: the toy/smoke benchmarks hand
every model the splitter's own windows through `run_backtest`, PatchTST's
move onto `_rv` changes validation and arithmetic only, and the CFD
retirement touches a symbol map nothing reads at fit time.

| # | Item | Verdict | Where / why |
|---|---|---|---|
| 1 | Index arithmetic at boundaries | **PASS** (one FIXED upstream) | `_patchtst_net._windows`: `x_i = y[i:i+L]`, `t_i = y[i+L:i+L+H]`, `i ≤ n-L-H`, so the last target is `y[n-1]` — the origin itself; the forecast input is `y[-lookback:]`. `lgbm._design_matrix`: rows `t = 21 .. n-2`, target `log rv[t+1]`, features `rv[t-21:t+1]`; `_feature_row` rejects `t < 21` (numpy's negative-start wrap). `panel.crisis_coverage` had re-derived the splitter's first scored position as `window + horizon` (it is `window`) — **FIXED on the panel branch**, now the union of the splitter's own `test` indices. |
| 2 | Splitter monopoly | **PASS** | Every model receives `train` from `run_backtest`, which slices `series` and `fit_series` with the same `Origin.train`; `benchmarks.toy` / `benchmarks.smoke_tsfm` construct one `RollingOriginSplitter` and nothing else. `data/panel.py` produces no train/test indices (its docstring says so; its trim is a calendar window on already-realized targets, not a split). The TSFM adapters take `rv[-cap:]` of the handed window, never of the full series. |
| 3 | Feature lags | **PASS** | LightGBM: 22 lags + HAR's weekly/monthly means over `rv[t-4..t]` / `rv[t-21..t]`, all ≤ t, from the handed window. HAR unchanged. `diagnostics.py` computes full-sample statistics *for the report* and nothing consumes them (the module says so; if one ever becomes a feature it must be recomputed per window). Crisis tags are dates only and structurally cannot reach a model (AST check in `tests/test_data_crisis.py`). |
| 4 | Transforms and scalers | **PASS** | LightGBM has no scaler (asserted: a window's design matrix is bit-identical whether the array continues afterwards). PatchTST's instance normalization is per input window (mean/sd of the 64 points being fed), not a fitted scaler. TSFM `input_scale` is a fixed constant. Smearing / Gaussian factors are estimated at the scheduled fit from that window's residuals and never re-estimated by `update`; statsforecast's ETS h-step variance is read from the scheduled fit, not from `forward` (which would re-estimate). `panel.repair_bars` tolerance is a constant. |
| 5 | Target construction | **PASS** | `overnight_variance` reads `O_t` and `C_{t-1}`; `rogers_satchell` reads day t's bar; the D-016 target is literally their sum. Panel targets are built on the full file history then trimmed, so day t's value depends on days ≤ t only (a corruption canary pins it). Inconsistent bars → NaN in place, never forward-filled, row kept. `parse_stooq_csv` sorts ascending and parses `YYYYMMDD` with an explicit format (a bare-integer read would land in 1970 and silently reorder). |
| 6 | Refit schedule | **PASS** | All three classical models implement `update` as re-conditioning at fixed parameters (statsforecast `forward`; LightGBM buffer refresh under fixed trees) with the smearing factor pinned at the fit; PatchTST is frozen between refits (`conditioned_through == fit_origin`); TSFM `update` is context extension. `smoke_tsfm` at `refit_every=21` changes no zero-shot number (`tests/test_smoke_tsfm.py`). |
| 7 | TSFM context windows | **PASS** | `_context_of` = trailing `min(max_context, context_length)` observations of the handed window, ending at the origin; one series per forward pass — no batching, padding or cross-series alignment exists yet. `FittedTSFM._cache` is per fitted object and `update` builds a new object, so a cached forecast for an older context can never serve a newer one. TimesFM re-'compiles' when the context length changes so a context is never padded against a stale maximum. |
| 8 | Calendar alignment | **PASS** | The panel never joins two assets; each `PanelSeries` stays on its own calendar. `inference.loss_matrix` aligns *models* on `origin_index` (a position on one cell's series) and refuses cells whose series bytes differ; it never aligns across assets. Crypto's target is 5-minute RV on the exchange's own day. |
| 9 | Caching | **PASS** | Every new cell's `config_hash` covers `array_digest` of `series`, `fit_series` and proxy (the classical models pass `fit_series`), the model `spec()` — which for the TSFMs includes the weights' resolved commit hash — the splitter and the seed. The 0.3.0 bump moves every hash so no 0.2.0 fragment can be served. `ingest_manual_csv` re-reads and re-hashes raw bytes every call; the Binance cache is keyed by immutable past days. |
| 10 | Survivorship & selection | **DESIGN-LEVEL, flagged** | The ten equity series are instruments liquid in 2026, chosen in 2026 (SPY/DIA/ISF explicitly for their history). Weaker for variance than for return studies, and index-level data carries reconstitution survivorship regardless. Belongs in the paper's data section (§10). |

**Canary.** Present and passing for every new code path: `tests/test_models_sf.py`,
`tests/test_models_lgbm.py`, `tests/test_models_patchtst.py`,
`tests/test_models_tsfm_common.py` each corrupt the series after a cutoff
through `run_backtest` and assert earlier rows bit-identical (with the
companion "corrupt earlier and the comparison fails" check where the stream
wrote one); `tests/test_data_panel.py::TestFutureDataCannotReachAnEarlierTarget`
does it for the panel's targets with `assert_frame_equal` and an inert-proof
companion; `tests/test_m1_smoke.py` covers the toy benchmark end to end, now
over eight models.

**One FIXED item not in the table's ten**, recorded because it is the shape of
a leak: `panel.repair_bars` read its classification arrays through
`to_numpy()`, which on a single-dtype frame is a *view*; clamping a repaired
bar rewrote them and a `high == low` day stopped being counted. Found on the
real NKX 2020-10-01 bar, regression test on the panel branch.


### 9.3 CI

Pushing `m2/p2-integration` triggered CI under the widened trigger.
Run 32794688768 (`462799e`): 3.11 green, 3.13 **failed** on the pinned
identities, 3.12 (see below). Run 32795672046 (`fb7ba7b`, with the config
dump): 3.12 **failed** the same way and the dump located the cause — §3.6.
Run 32795672046's other legs finished 3.11 **failed** / 3.13 passed — the
mirror image of run 32794688768 (3.11 passed / 3.13 failed) on the same
code, which is the runner lottery in one table. With the kernel-family pin
(`6bf3832`), run 32796535733: **3.11, 3.12, 3.13 all green**. The merge
commit on `main` gets its own run, which this file cannot carry; it is the
run tagged `v0.3.0-p2core`.


## 10. Flagged for a human — not resolved here

1. `docs/data_licenses.md` still presents the CFD substitution as an open
   question; D-012 answered it and D-024 removed the entries. The mirror
   should say so (planning machine).
2. D-012 is referenced by two streams and by this report but is not in the
   repo's `docs/decisions.md`; the D-021..D-026 numbers were provisional until
   the planning machine reconciles.
3. The paper's PatchTST numbers must state the device class (§3.4).
4. Survivorship in the panel (§5, data) belongs in the paper's data section.
5. `make reproduce` now takes ~1 minute longer and needs the `classical`
   extra; a machine without it fails by name rather than recording NaN rows.
   That is intended, but it is a change to what "reproduce" costs.


## 11. Follow-up — v0.4.0, the protocol decisions (`feat/p2-protocol`)

The three decisions §8 deferred, plus the ES gap the inference stream flagged,
shipped as one reviewed change against this merged tree. Recorded here because
this file is where a reader looks for "what happened after the integration".

### 11.1 What shipped

| | |
|---|---|
| **D-018** invalid-target policy | new `volbench.compaction` (`FitSeries`, `InsufficientHistoryError`); `PanelSeries.fit_input()` hands it out; `run_backtest` materializes each window through it |
| **D-019** fit window | `FIT_WINDOW_DEFAULT = 500`, `FIT_WINDOW_ROBUSTNESS = 1000` in `data/panel.py`; both are ordinary splitter windows, so both were already hashed |
| **D-020** panel list | `EQUITY_PANEL` is 9 equity series; ISF moves to `RETIRED_EQUITY` and stays ingestable; headline panel 11 assets |
| **ES columns** | `es_<level>` beside every `var_<level>` in the scored schema; `var_backtest` reads them with no argument |
| version | 0.3.0 → 0.4.0 |

**Where the ES arithmetic went, and why.** `evaluate.py` must not import
`backtests.py` — the dependency runs evaluation → results → distributions, and
the backtests read the evaluator's *output*. Both now need the same number, so
it went onto the object that owns it:
`volbench.dist.Distribution.expected_shortfall`, beside `variance` and `crps`,
with closed forms on `Normal`/`StudentT`, the exact integral of the
piecewise-linear quantile function on `Empirical`/`QuantileGrid`, and
Gauss-Legendre quadrature in the base class for anything else.
`backtests.expected_shortfall` stays as a public wrapper over it. Neither
consumer imports the other, and there is one implementation rather than two
that can drift.

`var_backtest(frame, level)` now scores FZ0 without an `es=` argument: the
column in the row is this cell's own ES at this level, from the same
distribution as its VaR, so it is the only ES that belongs in that loss. A
frame written before 0.4.0 has no such column and still backtests, with
`fz0_mean = None` as before. One related hardening: rows the missing-policy
already excluded now have their ES masked *before* the loss is evaluated, so a
degenerate ES on a dropped row can no longer fail FZ0's domain check for the
whole cell.

### 11.2 Pre-0.4.0 fragments are orphaned, and that is the mechanism working

`package_version` is in every `config_hash`, so **every hash moved at 0.4.0**
and no fragment written by 0.3.0 or earlier can be served again. That is
deliberate and it is the same argument as the 0.2.0 → 0.3.0 bump (D-021), only
stronger here, because two things changed that a stale row could not express:

- the **scored-path schema** gained `es_<level>` columns, so a 0.3.0 fragment
  is missing columns that 0.4.0 code expects to find;
- the **protocol semantics** changed: a variance-fed cell's fit windows are now
  compacted, so a 0.3.0 row for "the same" model and asset was produced from
  different training data wherever that series has an invalid day.

Orphaned means orphaned, not overwritten: `ResultsStore` fragments are written
once and never mutated, so the old files simply stop being addressed. Nothing
deletes them and nothing serves them. A store carried over from 0.3.0 will
recompute everything on first use, which for the toy benchmark is ~1 minute and
for a grid run is the whole run — worth knowing before starting one on a
populated store. The toy benchmark's pinned identities were regenerated once,
here (`tests/test_recondition.py`); its *numbers* did not move, because the toy
fixture has no invalid day for compaction to drop, which is asserted directly
rather than assumed.

### 11.3 Measured — what D-018 is worth on the real archives

HAR over the real HSI and TWSE series, window 500, `refit_every=21`, both arms
of the policy on the same data:

| series | days | invalid days | policy | origins with a forecast | flagged rows |
|---|---|---|---|---|---|
| HSI | 5328 | 13 (12 zero, 1 NaN) | `compact` | **4828 / 4828 (100%)** | 13 |
| HSI | 5328 | 13 | `none` | 3066 / 4828 (63.5%) | 1765 |
| TWSE | 5301 | 80 (all NaN) | `compact` | **4801 / 4801 (100%)** | 80 |
| TWSE | 5301 | 80 | `none` | 2326 / 4801 (48.4%) | 2477 |

Uncompacted, 13 unusable days cost HSI a third of its HAR column and 80 cost
TWSE more than half — as correctly-recorded NaN rows, which is why it was
invisible. Compacted, the only unscorable rows are the ones whose *own target*
is unmeasurable: exactly 13 and 80, and no others.

### 11.4 Gates

Filled in from the runs on `feat/p2-protocol`.

- `ruff check .` and `mypy --strict` — clean on 3.11 / 3.12 / 3.13.
- Full CPU suite on 3.11 / 3.12 / 3.13.
- `make reproduce` at 0.4.0: green, and byte-identical against the 0.4.0
  baseline committed on this branch.
- tsfm / gpu opt-in suites on the 4090.
- Leakage check over the full diff, with compaction as the focus.
- CI on the branch push (the widened trigger covers `feat/**`).

See §12 for the numbers.


## 12. Gates and leakage audit for v0.4.0

### 12.1 Local runs

All on this box (RTX 4090, driver 535, torch 2.5.1+cu121 in the 3.11 venv;
3.12/3.13 venvs built with the CI extras), with the kernel-family pin
(D-026) exported.

| Leg | Extras | ruff | mypy --strict | pytest |
|---|---|---|---|---|
| 3.11 | classical + tsfm | clean | clean, 41 files | **1042 passed, 29 skipped**, 7 m 24 s |
| 3.12 (`CI=true`) | classical + torch-cpu | clean | clean | **1037 passed, 35 skipped**, 7 m 19 s |
| 3.13 (`CI=true`) | classical + torch-cpu | clean | clean | **1037 passed, 35 skipped**, 7 m 04 s |
| 3.11 opt-in (`VOLBENCH_RUN_TSFM=1 VOLBENCH_RUN_GPU=1 -m "tsfm or gpu or timegpt"`) | classical + tsfm | — | — | **29 passed, 1 skipped** (TimeGPT: no `NIXTLA_API_KEY`, by design) |

The suite grew by 108 tests, 43 of them the new `tests/test_compaction.py`.
The 3.12/3.13 legs ran five tests fewer because they were collected before the
last commits; CI re-ran the whole matrix on the final tree, which is the
authoritative matrix result (§12.3). The larger skip count on 3.12/3.13 is the
`tsfm`-marked set, which `CI=true` never honours.

Two of the 3.11 tests do not run anywhere else, by design:
`test_the_real_panel_series_keep_every_origin` re-checks D-018 on the **real**
HSI and TWSE archives (1.4 s each) and skips wherever they are not unpacked —
and always under `CI`, since they are hand-downloaded and never committed. The
mechanism-faithful fixtures in the same file carry the contract for CI; this
one keeps §11.3's numbers re-runnable rather than quoted.

`make reproduce` at 0.4.0: green, and the eight fragments **byte-identical**
across two from-scratch rebuilds (results directory deleted, fixture
regenerated and diffed clean against the committed CSV). `make smoke-tsfm`
likewise: four fragments byte-identical across two runs on the 4090.

Fragment identities at 0.4.0 (`tests/test_recondition.py` pins the config
hashes; the parquet digests are recorded here so a future rebuild can be
compared without re-deriving them):

```
autoarima b9ec1db0…0da75a3e   autoets 48eb0390…035348aa   ewma 8c4fe260…3514cb0f
garch11   508a70e0…f78b37a0   garch11_t 50ae4977…c1dfcfe9  har  d793f8ce…6d080c0427
lgbm      4edaa744…c20c9fd838 naive   109e1f8f…0062efa1
smoke-tsfm: chronos 1e09792e…6c30d18  moirai 9ccb5a09…ce07f4e67
            patchtst d4611f30…eebe8a1e timesfm ed8975f4…5771f23ed
```

The eight toy scores are unchanged to every printed digit from the 0.3.0 run
(§7.1), and the four smoke-tsfm scores from §7.2 — only the hashes moved,
with the version and the recorded policy. That is the expected result and it
is the point: compaction is a no-op on a fixture with no invalid day, and the
version bump is what orphans the old fragments.

### 12.2 Leakage check (`.claude/skills/leakage-check`, `main...feat/p2-protocol`)

Run over the full diff (19 files) with compaction as the focus, since it is
the one thing in this project's history that *rewrites which past
observations a training window contains*.

| # | Item | Verdict | Where / why |
|---|---|---|---|
| 1 | Index arithmetic at boundaries | **PASS** | `compaction.window_positions`: `available = searchsorted(valid_positions, origin, side="right")` counts valid positions `<= origin` (`side="right"` keeps the origin's own observation, which is available at t), then returns `valid_positions[available - n : available]`, whose maximum is therefore `<= origin`. Bounds are validated first: `train` strictly increasing, `positions[0] >= 0`, `origin < size`. `dropped_positions` spans `[kept[0], kept[-1]]`, so every dropped day is `< origin`. Swept over 400+ origins of a doubly-defective series by `test_every_window_ends_at_or_before_its_origin` and `test_every_dropped_day_lies_strictly_in_the_past_of_its_origin`. |
| 2 | Splitter monopoly | **PASS** (one array worth naming) | `evaluate.py` calls `model.fit(task.fit_series.window(origin.train))` and `fitted.update(task.fit_series.window(origin.train))` — the splitter's own array, passed through; `window_positions` reads only `positions.size` and `positions[-1]`, i.e. exactly the "N observations ending at t" pair the splitter guarantees. The one array that *looks* like index production is `FitSeries.valid_positions = flatnonzero(valid_target_mask(values))`. It is not a train/test index: it is a validity index over the whole series, and it is only ever intersected with "`<= origin`". `test_the_splitter_still_runs_on_the_full_calendar` pins that origins and targets are unchanged by the policy. |
| 3 | Feature lags | **PASS**, with a documented semantic change | Every value in a materialized window sits at a calendar position `<= origin`. `har._design_matrix`/`_har_features` and `lgbm._design_matrix`/`_feature_row` are untouched and read only the handed array. What changed is what a lag *means*: position `p-1` is the previous **measured** day, so a lag-1 regressor can span two or more calendar days. Documented in both adapters, in `compaction.py`, in `data/panel.py` and in `docs/design.md`; pinned by `test_a_dropped_day_makes_the_lag_span_two_calendar_days`. |
| 4 | Transforms and scalers | **PASS** | Nothing new is fitted. `valid_target_mask` is a *pointwise* predicate (`isfinite & > 0`), not a statistic estimated from the series, so computing it over the whole array transports no information — and selection is bounded by the origin regardless. `test_validity_of_a_later_day_cannot_change_an_earlier_window` makes every day after an origin invalid and shows that origin's window unchanged. Smearing/Gaussian factors and the TSFM `input_scale` are as before. |
| 5 | Target construction | **PASS** | `repair_bars` and `build_targets` are untouched; `PanelSeries.fit_input` reads an already-built column. The scored path is unchanged, so day t's target still reads day t's own bar plus day t-1's close. Nothing is forward-filled: an invalid day is dropped from *windows*, never replaced by a neighbour — which is exactly why option (d) was rejected in D-018. |
| 6 | Refit schedule | **PASS** | Window materialization happens inside the `fit`/`update` it belongs to, so it is bounded by that call's own origin. A short window at a *scheduled* fit fails its whole refit block, the same rule as any other fit failure — never an off-schedule refit, which would report a cadence the run did not use. Pinned by `test_a_short_scheduled_fit_fails_its_whole_block_like_any_other`; structurally this can only bite a series' first block. |
| 7 | TSFM context windows | **PASS** | `_context_of` still takes the trailing `min(max_context, context_length)` of the handed window, which now ends at the last valid observation `<= origin`. All four zero-shot adapters and PatchTST are `fits_on_variance=True` in `smoke_tsfm`, so they receive the same compacted object as HAR; still one series per forward pass, so no padding or cross-series alignment exists. |
| 8 | Calendar alignment | **PASS** | `FitSeries` carries the pandas index it was built from precisely so `_require_one_calendar` keeps working: values and order are still checked, an off-calendar fit series still raises, and a *bare-array* `FitSeries` beside indexed inputs is still refused rather than silently aligned by position. Both pinned (`test_a_fit_series_off_calendar_is_still_refused`, `test_a_bare_fit_series_beside_indexed_inputs_is_refused`). No cross-asset join exists anywhere. |
| 9 | Caching | **PASS** | `array_digest(fit_input.values)` covers the full-calendar values, NaNs included — numpy's canonical quiet NaN, verified identical across the construction paths that produce it. `protocol.invalid_target_policy` is recorded whenever it is not `"none"`, so the two arms cannot share a cache entry (`test_the_two_arms_can_never_share_a_cache_entry`), while a bare array keeps the pre-D-018 identity exactly (`test_a_bare_array_means_no_policy_and_hashes_as_it_always_did`). The 0.4.0 bump moves every hash, so no pre-0.4.0 fragment is served. |
| 10 | Survivorship & selection | **DESIGN-LEVEL, flagged** | Unchanged in kind, sharper in degree. D-020 removes ISF from the panel using a fact knowable only in 2026 — that the fund launched in 2015. That is selection on *data availability*, not on outcomes, so it cannot bias a forecast comparison; but the panel's composition is now explicitly the product of a 2026 judgement about history length, on top of the SPY/DIA choice §9.2 already flagged. Belongs in the paper's data section, stated plainly. |

**Canary.** Strengthened, not just carried over. `tests/test_compaction.py`
corrupts (a) every *value* after an origin and (b) every later day's
*validity* — the subtler case, since compaction reads validity — and asserts
the window is bit-identical either way, with an inert-proof companion that
corrupts from *before* the origin and requires the comparison to fail. The
end-to-end canary in `tests/test_evaluate.py` now also compares the
`es_<level>` columns, and `tests/test_m1_smoke.py` is unchanged.

**One FIX, applied during the audit.** `run_backtest`'s docstring stated the
guarantee as "*volbench itself passes a model nothing but
`fit_series[origin.train]`*", which compaction makes false. Corrected to the
bound that is actually true — nothing but observations at positions `<=
origin.origin`, either `fit_series[origin.train]` exactly or the last
`len(origin.train)` valid ones of them. Documentation rather than behaviour,
but a stale invariant statement in the one place a reader looks for it is how
a real invariant gets eroded.

### 12.3 CI

Pushing `feat/p2-protocol` triggers the matrix under the widened trigger
(`feat/**`). Run recorded at merge time.
