# P3 analysis — what the store actually is

**Scope.** Eight assumptions the analysis layer was specified against, each
checked against the repository and the completed primary grid, and the actual
state written down where they differ. Later analysis prompts read this file
instead of re-deriving it.

**Reported, not interpreted.** Nothing here reads a score as evidence about a
model. Where a number appears it is a property of the *store*, not of a
forecast.

| | |
|---|---|
| Package | volbench 0.6.0, branch `feat/p3-analysis` (from `99e3ea4`) |
| Grid | 143 cells = 11 assets x 13 configs x h=1 x 1 arm; 645,151 rows |
| Store | `data/grid_primary/store/` — 143 parquet fragments + 143 JSON sidecars |
| Manifest | `docs/P3_GRID_manifest.json` (byte-identical to `data/grid_primary/manifest_primary.json`) |
| Analysis entry point | `src/volbench/analysis.py`; `tests/test_analysis.py` |
| Checked on | 2026-08-28 |

---

## Summary

| # | Assumption | Verdict |
|---|---|---|
| 1 | Long/tidy frame keyed by (asset, config, horizon, arm, origin/target, **date**) | **Partly wrong** — no date column, no arm column; key is `(config_hash, asset, origin_index, horizon)` |
| 2 | Per-row losses stored or recomputable: CRPS, log score, pinball, QLIKE, FZ0 | **Confirmed, with a split** — four stored, FZ0 recomputable exactly |
| 3 | VaR/ES per row recoverable or stored | **Confirmed** — stored, at 0.01 / 0.025 / 0.05, lower tail |
| 4 | `docs/` holds the manifest with config hashes, thread pins **and data digests** | **Partly wrong** — hashes and pins yes; **digests are not committed anywhere** |
| 5 | Convergence/fallback recorded per fit, fit origin recoverable | **Confirmed exactly** — and only for 3 of 13 configs (docs/P3_INSTRUMENTATION_GAP.md) |
| 6 | `econ.py` exists, imports nothing from the package | **Confirmed** |
| 7 | Named crisis windows in the codebase | **Confirmed** — 4 dated + 1 deliberately undated; **not** the fallback dates |
| 8 | Which interpreter ran the grid | **Recorded here for the first time** — CPython **3.11.5** |

---

## 1. The results frame — key, columns, and the two things that are not there

**Assumption.** "reads back into a long/tidy frame keyed by (asset, config,
horizon, protocol arm, origin/target_index, date), one row per scored day,
including NaN-score rows carrying a `missing_reason`."

**Actual.** Long/tidy and one row per scored day: yes. NaN rows carrying a
`missing_reason` rather than being dropped: yes, and that is the contract
(`evaluate._score`). The key is different, and two of the six named fields do
not exist.

The uniqueness key is `results.KEY_COLUMNS`:

```
(config_hash, asset, origin_index, horizon)
```

`ResultsStore.write` refuses a frame with a duplicate on it. Every fragment's
32 columns:

| group | columns |
|---|---|
| identity | `config_hash` `asset` `model` `origin_index` `horizon` `target_index` `seed` |
| protocol trace | `fit_origin` `conditioned_through` `refit` `fit_status` |
| forecast | `forecast_mean` `forecast_var` `var_{0p01,0p025,0p05}` `es_{0p01,0p025,0p05}` |
| target | `realized_return` `proxy_name` `proxy_var` |
| losses | `crps` `log_score` `qlike` `pinball_{0p01,0p025,0p05}` |
| indicators | `hit_{0p01,0p025,0p05}` |
| accounting | `missing_reason` |

**No `date` column.** Rows carry `target_index`, an integer position into the
series the cell was run on. Dates are recoverable only by rebuilding the panel
(`build_panel()`, ~63 s) and applying the driver's leading trim. This matters
for the crisis sub-samples of §7, which are defined by dates: the join is
`results.target_index -> panel position -> timestamp`, and it is an extra step
with an off-by-one in it (§4 of docs/P3_ANALYSIS_VALIDITY.md).

**No protocol-arm column.** The arm is a property of the *cell*, and it lives
in the manifest (`cells[].arm`), not in the fragment. The arm's *settings*
reach the fragment only through `config_hash`, via the splitter's fields and
the `protocol` block of the sidecar. `analysis.load_grid` joins the manifest's
`arm` and `lane` onto the rows so downstream code has them; it does not invent
them. One arm exists today (`headline`), so nothing is currently ambiguous —
but a two-arm grid read without that join would silently pool the arms.

**`model` is the adapter's name, not the grid's label.** The fragment's `model`
column holds e.g. `garch(1,1)-studentst`; the manifest's `model` field holds
`garch11_t`. They are 1:1 across the 13 configs, checked. `load_grid` keeps
both and adds the manifest's spelling as `model_label`.

## 2. Losses — four stored, one recomputable, one absent by design

| loss | stored? | how it is obtained |
|---|---|---|
| CRPS | **stored** (`crps`) | `Distribution.crps` at scoring time |
| Log score | **stored** (`log_score`) | `Distribution.log_score` |
| QLIKE | **stored** (`qlike`) | `metrics.qlike(forecast_var, proxy_var)` |
| Pinball | **stored** at 0.01/0.025/0.05 | `Distribution.pinball` |
| FZ0 joint (VaR, ES) | **not stored** — recomputable exactly | `backtests.fz0_loss(realized_return, var_a, es_a, a)` from four stored columns |
| MSE on the variance scale | not stored — recomputable | `(forecast_var - proxy_var)^2` |
| VaR hits | **stored** (`hit_*`) | `1{realized_return < var_a}` |

FZ0's domain was checked over the whole grid before claiming it is computable:
of the 644,983 rows carrying a finite (VaR, ES) pair at each of the three
levels, **zero** violate `e < 0`, `e <= v`, or `v < 0`. The remaining 168 rows
have no forecast at all (§2 of docs/P3_ANALYSIS_VALIDITY.md), so FZ0 is
defined wherever a forecast exists.

The three losses that are stored *and* recomputable were recomputed from
independent closed forms over all 645,151 rows as the alignment canary;
agreement is at machine precision. See docs/P3_ANALYSIS_VALIDITY.md §4.

## 3. VaR and ES — stored, and the predictive law is recoverable

**Levels actually evaluated: 0.01, 0.025, 0.05** — `evaluate.DEFAULT_LEVELS`,
recorded in every sidecar as `scoring.levels`, identical across all 143 cells.
Lower tail only, in the return-side sign convention (negative). Both
`var_<level>` and `es_<level>` are written on every row where a forecast
exists, *including* rows whose target is unscorable — a forecast that was made
is described whether or not it could be scored.

The `Distribution` object itself is **not** persisted. It is nevertheless
recoverable from the stored columns, because of what the adapters emit:

- twelve of the thirteen configs emit `Normal(0, sqrt(forecast_var))` over the
  next-period return — verified over the grid, not assumed: the stored
  `var_0p01`, `var_0p025` and `var_0p05` reproduce the Gaussian quantiles of
  `(forecast_mean, forecast_var)` to a maximum absolute error of 1.7e-16 on all
  595,524 rows of those twelve;
- `garch11_t` emits a location-scale Student-t at the same variance whenever
  its own estimator ran, and a `Normal` on the 18 fits that fell back to EWMA.
  Its `nu` is not stored but is recoverable by inverting the *ratio* of two
  stored tail quantiles (`analysis.student_t_df_from_quantile_ratio`).

All 2,367 `garch11_t` scheduled fits recover: 2,349 Student-t with `nu` in
[2.264, 50.000], 18 Gaussian — exactly the 18 the `fit_status` column labels
`fallback=ewma`. The recovered `nu` is identical across every origin of every
one of the 2,349 blocks, which is independent confirmation from the stored
artifacts that `update` re-conditions without re-estimating.

## 4. What `docs/` holds — and the digests it does not

| item | in `docs/`? | where it actually is |
|---|---|---|
| Config hash per cell | **yes** | `P3_GRID_manifest.json` -> `cells[].config_hash` |
| Thread pin | **yes** | `P3_GRID_manifest.json` -> `environment.blas_threads = 1`, `thread_pin_explicit`, `env.*` |
| Kernel pin / signature | **yes** | `environment.kernel_signature`, `environment.env.NPY_DISABLE_CPU_FEATURES` |
| BLAS build and observed pools | **yes** | `environment.blas`, `environment.observed_thread_pools` |
| **Data digests** | **no** | only in the 143 JSON sidecars under `data/grid_primary/store/`, which are gitignored |

`series_sha256`, `fit_series_sha256`, `proxy.sha256` and `raw_sha256` are what
tie a cell's numbers to specific bytes of input data — the leakage control
`results.py` describes in terms. The manifest contains no `sha256` field at
all: searched, none. From a clean checkout the committed record therefore names
*which experiment* each cell is (its hash) but not *what data* it ran on.

This is the same class of gap as the uncommitted driver (§4 of
docs/P3_DRIVER_PROVENANCE.md) and is flagged, not fixed: writing digests into a
committed file is a change to what a run publishes, and belongs to whoever owns
the manifest format.

Also absent from the manifest and from every sidecar: the **Python version**
(§8), and the versions of `numpy`, `scipy` and `pandas`. The four torch-backed
adapters do record their own backend versions inside `model.spec()`, and those
are hashed.

## 5. Convergence and fallback — per fit, origin recoverable, three configs

**Confirmed, exactly as stated.** `fit_status` is written on every row but
describes the *scheduled fit at* `fit_origin`, not the origin of the row: the
21 origins of a refit block all carry the status of the one fit they rest on.
Checked rather than assumed —

- 7,101 scheduled fits carry a reported status;
- **0** of them have a `fit_status` that varies across the rows resting on it;
- **0** rows carry a non-empty `fit_status` with `fit_origin == -1`.

So `groupby(["config_hash", "fit_origin"])["fit_status"].first()` is the
per-fit view, and the fit's origin is `fit_origin`. Recounting that way
reproduces the manifest exactly: 7,101 fits, 38 fallbacks, 38 non-converged.

| config | fits | fallback | non-converged |
|---|---:|---:|---:|
| `garch11` | 2,367 | 4 | 4 |
| `garch11_t` | 2,367 | 18 | 18 |
| `gjr` | 2,367 | 16 | 16 |
| **the other ten** | **0** | — | — |

The vocabulary in the store is `""` (10 configs, every row), `ok`, and
`fallback=ewma|flag=<n>`; `nonconverged` never appears as a head, because for
these adapters a non-convergence is what causes the fallback. Empty is
reserved for "this model reports nothing" and never means a clean fit —
`FitDiagnostics.status()` cannot return it. The consequences are §3's subject:
docs/P3_INSTRUMENTATION_GAP.md.

## 6. `econ.py`

**Confirmed.** `src/volbench/econ.py` (480 lines) implements the
volatility-targeting backtest and imports **nothing** from `volbench` — only
`math`, `warnings`, `dataclasses`, `typing`, `numpy`, `pandas`. Its import
graph is pinned by `tests/test_econ.py::TestBoundary::test_it_does_not_import_the_evaluator_or_any_model`.

`src/volbench/analysis.py` is held to the same shape by
`tests/test_analysis.py::TestBoundary`, with one deliberate difference: it may
import `volbench.results`, because reading the store *is* its job, and
`ResultsStore` is the sanctioned way to address a fragment. Everything that
could fit a model, cut a window or run a cell is denied by name
(`volbench.models`, `.evaluate`, `.runner`, `.execute`, `.splitter`,
`.benchmarks`, `.compaction`), the allowed set is an allow-list rather than
only a deny-list, and `volbench.dist`/`volbench.metrics` are denied too so that
the loss recomputation in §5 of the validity report stays independent of the
code that produced the numbers it checks.

## 7. Crisis windows — the codebase defines them; use these, not the fallback

`src/volbench/data/crisis.py` defines named windows, so per the instruction
they are used verbatim. **They are not the fallback dates**: the fallback GFC
window (2007-07-01 -> 2009-06-30) is 24 months; the codebase's is 7.

| tag | start | end | label | source phrase (research_design.md) |
|---|---|---|---|---|
| `gfc` | 2008-09-01 | 2009-03-31 | Global financial crisis | "GFC Sep 08-Mar 09" |
| `covid` | 2020-02-01 | 2020-04-30 | COVID-19 crash | "COVID Feb-Apr 20" |
| `tightening_2022` | 2022-01-01 | 2022-10-31 | 2022 monetary tightening | "2022 tightening Jan-Oct 22" |
| `spike_2024_08` | 2024-08-01 | 2024-08-31 | August 2024 volatility spike | "Aug-2024 spike" |

Only `covid` coincides with the fallback. A fifth window is **named and
deliberately undated**:

| tag | status |
|---|---|
| `stress_2025_26` | `PENDING_WINDOWS` — D-004 fixes its dates at grid freeze; `window_by_tag` raises rather than guessing |

Everything outside all four is tagged `calm` (`crisis.CALM_TAG`), a real string
rather than NaN so a groupby cannot silently drop it. The windows do not
overlap (asserted at import by `tests/test_data_crisis.py`), `tag_dates`
returns an ordered Categorical, and `crisis_table` produces mutually exclusive,
jointly exhaustive columns.

`spike_2024_08` carries a caveat in its own source: research_design.md gives
only the month, and the module records that the day-level range was read as the
calendar month containing the 2024-08-05 unwind, flagged for confirmation.

`crisis.py` needs a **tz-aware DatetimeIndex** and raises on a naive one. Since
the results frame has no dates (§1), tagging requires the panel rebuild.

## 8. The interpreter that ran the primary grid

**CPython 3.11.5**, in the repository's default `uv` project environment
`.venv`.

Not recorded anywhere by the run itself — not in the manifest, not in any
sidecar, not in the config hash. Established here by three independent facts,
because the instruction is right that it belongs in the record rather than in
memory:

1. `.venv` is the only environment of the three in the tree that carries the
   `tsfm` extra (`torch 2.5.1+cu121`, `chronos-forecasting`, `timesfm`,
   `uni2ts`). `.venv-py3.12` (3.12.14) and `.venv-py3.13` (3.13.15) carry
   `torch 2.5.1+cpu` and none of the foundation-model backends. The grid ran a
   44-cell GPU lane, so it cannot have run in either.
2. The versions the sidecars record inside `model.spec()` match `.venv`
   exactly and match neither other environment: `torch 2.5.1+cu121`,
   `transformers 5.15.1`, `chronos_forecasting 2.3.1`, `timesfm 2.0.2`,
   `uni2ts 2.0.0`.
3. `UV_PROJECT_ENVIRONMENT` is unset, so `uv run` resolves to `.venv`, which is
   how docs/P3_GRID.md's invocation line was run.

The rest of `.venv`, for the record: `numpy 2.4.6`, `scipy 1.17.1`,
`pandas 3.0.5`, `arch 8.0.0`, `statsforecast 2.1.1`, `lightgbm 4.7.0`,
`volbench 0.6.0`.

**Why this is worth a paragraph.** The 3.12 `getattr_static` protocol defect was
established as test-double-only and production `fit_status` was verified
unaffected — but `fit_status` is the exact column that produced the 38
fallbacks of §5, so the interpreter is part of that column's provenance. It is
now recorded: the grid did not run on 3.12 at all, so the defect could not have
reached it by any route, sound or unsound.

Two drift notes, flagged not acted on:

- The analysis stream's brief says Python 3.12+; `pyproject.toml` says
  `requires-python = ">=3.11"`, ruff targets `py311`, CI covers 3.11-3.13, and
  the grid ran on 3.11.5. `src/volbench/analysis.py` is written to run on all
  three and is checked on 3.11.5.
- `volbench.analysis` is **not** exported from `volbench/__init__.py`. The
  root namespace is a public API surface whose record is `docs/design.md`, and
  that file is a read-only mirror here (CLAUDE.md), so widening the root is a
  decision for the planning machine. `import volbench.analysis` works.
