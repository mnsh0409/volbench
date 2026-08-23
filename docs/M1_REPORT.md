# M1 integration report — volbench v0.1.0-m1

**Date:** 2026-08-23 · **Branch:** `main` · **Milestone:** M1 (due 27 Aug 2026)

Three Phase 1 streams (A data, B models, C evaluation) merged into `main` and
closed. All three were green on their own branch before merging and merged
with **zero conflicts** — the disjoint-file-ownership rule in
`docs/phase1_prompts.md` held exactly as intended. Every disagreement this
report describes is a *semantic* one that only became visible once the three
were composed.

**State:** 294 tests green; `ruff` and `mypy --strict` clean on `src` (3,395
lines) plus the interface test; `make reproduce` rebuilds the toy benchmark
from scratch.

---

## 1. What exists

### Stream A — data (`volbench.data`, 7 modules, 66 tests)

`TimeSeriesFrame` (validating OHLC container, UTC, no NaN, strictly
increasing); four variance proxies (`squared_return`, `parkinson`,
`garman_klass`, `realized_variance_from_bars`) all in daily units; a Stooq
daily-OHLC downloader with manual-CSV ingestion; a Binance 1-min-to-daily-RV
pipeline; a bring-your-own-data loader. `docs/data_licenses.md` records the
licence position of every source actually implemented.

### Stream B — models (`volbench.models`, 6 modules, 54 tests)

One Protocol pair (`ForecastModel`/`FittedModel`) and four baselines:
`NaiveVol`, `EWMA(λ=0.94)`, `GARCH`/`gjr_garch` (normal and Student-t, via
`arch`, with an EWMA fallback on non-convergence), and `HAR` (HAR-RV in logs
with a lognormal retransformation). All four honour the return-distribution
convention: `predict(h)` returns a `Distribution` over the next-period return.

### Stream C — evaluation (`volbench.evaluate`/`results`/`execute`, 86 tests)

`run_backtest` over `RollingOriginSplitter` origins, scoring CRPS, log score,
QLIKE, and pinball/VaR/hit at {1%, 2.5%, 5%}; a content-addressed
`ResultsStore` (append-only parquet, one fragment per `config_hash`, cache
short-circuit); and an `Executor` seam with a serial reference backend.

### Added at integration (this session)

- `volbench/__init__.py` — the wired public surface (43 exports, sorted).
- `data.log_returns()` — the missing A↔C seam; the data layer exposed only
  `r²`, but models and scoring both need *signed* returns.
- `volbench.benchmarks` — the toy benchmark and its fixture generator.
- `tests/test_model_interface.py` (21 tests) and `tests/test_m1_smoke.py`
  (16 tests, including the leakage canary).
- `py.typed` — the package was fully annotated but did not advertise it, so
  downstream type-checkers saw it as untyped.

---

## 2. The toy benchmark — measured

Four baselines × 200 rolling origins (window 500, h=1, step 1, refit every
origin) on a synthetic daily series, scored against the Parkinson proxy.

| stage | measured |
|---|---|
| compute only, in-process | **1.26 s** (1.26 / 1.55 / 1.25 over three runs) |
| full CLI run incl. interpreter + imports + parquet | **2.14 s** (2.15 / 2.08 / 2.19) |
| peak RSS | 197 MB |
| `make reproduce` end to end (check suite + rebuild) | **18.4 s** |
| **budget** | **120 s** |

Roughly 55× inside the two-minute budget. Determinism verified two ways:
identical config hashes across runs, and byte-identical parquet fragments.

```
| label   | model             |  n  | crps       | log_score | qlike    | mean_vol  | hit@1% | hit@2.5% | hit@5% |
|---------|-------------------|-----|------------|-----------|----------|-----------|--------|----------|--------|
| har     | har_rv            | 200 | 0.00762735 | -2.86267  | 0.215548 | 0.0127488 | 0.010  | 0.040    | 0.065  |
| ewma    | ewma              | 200 | 0.00762888 | -2.87887  | 0.203499 | 0.0137067 | 0.010  | 0.025    | 0.055  |
| garch11 | garch(1,1)-normal | 200 | 0.00765655 | -2.86383  | 0.217731 | 0.0127962 | 0.015  | 0.035    | 0.070  |
| naive   | naive_rw_vol      | 200 | 0.00767797 | -2.8005   | 0.329957 | 0.0112003 | 0.040  | 0.050    | 0.070  |
```

**These numbers are a wiring signal and nothing else.** The series is
synthetic, n=200, and no inference was run. The only thing worth reading off
it is that the pipeline behaves sanely: `naive` is clearly worst on QLIKE and
badly under-covers the 1% tail (4% hits against 1% nominal), while the
conditional models sit near nominal. CRPS barely separates them, which is
expected — CRPS on a *return* distribution is dominated by the return itself,
not by the variance forecast.

### Why the fixture is synthetic

The brief allowed "a committed fixture if the network is unavailable". The
network was **not** the binding constraint — the licence was:

- **Stooq**: redistribution is an explicit NO (ToS §5.3; §6.1 restricts the
  S&P DJI series further to personal, non-commercial use). Re-verified at
  integration that the CSV endpoint still answers with its JS proof-of-work
  anti-bot page rather than data. volbench does not attempt to bypass it.
- **Binance**: the archive is reachable (verified, HTTP 200), but whether the
  *derived* daily-RV series may be redistributed is unconfirmed pending a
  human read of the full Terms of Use.

CLAUDE.md forbids vendoring data that is not clearly redistributable, and
stream A's brief forbids tests that reach the network. A real asset therefore
could not be committed. The fixture is generated by
`benchmarks/make_toy_asset.py` (GARCH(1,1) returns, Brownian-bridge intraday
path so the high/low are consistent with each day's own close) and `make
reproduce` fails if regenerating it disagrees with what is committed.

One fixture bug was found and fixed during this session: at 13 intraday steps
the simulated high/low sat inside the true range of the path, biasing the
Parkinson proxy ~30% low and making HAR look like it under-forecast tail risk
(1% hit rate 0.050). At 390 steps — one per minute of a 6.5-hour session — the
bias is ~4% and the table reads correctly. Range estimators assume a
continuously-observed path; that is a general trap, not just a fixture detail.

---

## 3. Where each stream deviated from its brief

### Stream A

- **Stooq symbols are not the ones D-004 names.** `^spx`, `^dji` and `^ftse`
  have been retired by Stooq and now resolve to *CFD proxy instruments*
  (`^uslc` "U.S. Large Cap CFD", `^usbc`, `^uklc`). The stream updated
  `STOOQ_INDEX_SYMBOLS` to symbols that exist and flagged the substitution
  rather than silently absorbing it. **This needs a human decision** — a CFD
  tracking an index is not the index (see §6).
- **`download_index` is effectively non-functional**; `ingest_manual_csv` is
  the supported path. Correct call given the anti-bot gate, but it means the
  D-004 panel cannot currently be assembled without manual browser downloads.
- No `DataAdapter` protocol was built; the three sources have source-shaped
  signatures with nothing in common.

### Stream B

- **`HAR.fit` takes a realized-variance series, not returns** — documented
  clearly, and the evaluator supports it via `fit_series`, but it means the
  model interface is uniform in *type* and not in *meaning*.
- **Student-t GARCH returns a 199-point `QuantileGrid`**, not a parametric
  distribution. Well-reasoned (no RNG ⇒ bit-identical scores), but it
  interacts badly with the evaluator — see §4.2.
- **No model implements `SupportsUpdate`** (stream C's optional
  re-conditioning hook). Stream B could not have known: the hook was designed
  in the parallel stream.
- Inconsistent failure policy: `GARCH.fit` never raises (falls back to EWMA
  and records it); `HAR.fit` raises on degenerate input.

### Stream C

- **`Evaluator` became a function, not a class.** `run_backtest` scores a
  whole cell rather than consuming a `(Distribution, target)` stream.
- **DM / MCS are not implemented.** The comparison-inference half of the
  design's `Evaluator` — the tool that answers "who wins" — is Phase 2.
- **Added `SupportsUpdate` and `forecast_moments`** beyond the brief.
- **Hashes data *content*, not just a data label.** A genuine improvement: a
  cached artifact computed from a revised series can never be served for this
  one.
- Defined local `ForecastModel`/`FittedModel` Protocols as instructed —
  reconciled at integration (§4.1).

---

## 4. Cross-stream disagreements surfaced at integration

This is what the session existed to find. None of these were visible from
inside a single stream.

### 4.1 The model interface — RESOLVED

Stream C's local `FittedModel` required only `predict(h)`; stream B's requires
`name`, `spec()` and `predict(h)`. B honours the return-distribution
convention in all four models, so per the brief the reconciliation went **C →
B**: `evaluate.py` now imports the one definition from `models/base.py` and
re-exports it. `tests/test_model_interface.py` asserts the two are the *same
object*, statically under mypy and at runtime, so a second copy cannot
reappear silently.

### 4.2 QLIKE is biased for Student-t GARCH — OPEN

B returns Student-t forecasts as a `QuantileGrid` spanning τ ∈ [0.005, 0.995].
C's `forecast_moments` computes the exact variance *of that grid's law*, which
has flat tails — so it truncates the t-distribution's tails and understates
the variance. Both decisions are defensible alone; together they bias QLIKE.

Measured, for a *perfectly specified* forecast scored against a *perfect*
proxy (QLIKE should be exactly 0):

| ν | variance understated by | QLIKE floor |
|---|---|---|
| 3 | 23.8% | 0.0407 |
| 5 | 7.9% | 0.0035 |
| 8 | 4.3% | 0.0010 |
| 20 | 2.4% | 0.0003 |

A Student-t GARCH cannot score 0 on QLIKE no matter how right it is, and the
penalty grows exactly as the tails get heavier — i.e. precisely where the
Student-t specification is supposed to win. The toy benchmark uses normal
innovations so it does not hit this, but `docs/research_design.md` lists
Student-t variants in the 13-config model set. **This will silently distort a
headline result if it reaches Phase 3.**

### 4.3 The refit schedule does not mean what the protocol says — OPEN

`docs/research_design.md` specifies "refit every 21 trading days". Stream C
built `SupportsUpdate` so a model could re-estimate every 21 days while
re-filtering its conditional variance *daily* — the standard reading. No
stream B model implements it, so at `refit_every=21` a GARCH forecast is
**frozen for 21 days**, ignoring three weeks of realized returns.

It is not silent — every row records `conditioned_through` — and it is not
leakage (it errs toward *less* information). But it is not the protocol the
research design describes, and it would make the reported refit cadence
misleading. The toy benchmark therefore runs at `refit_every=1`, the only
honest setting today. Not fixed here: implementing `update()` changes forecast
numbers and is a methodological choice that deserves its own decision record,
not an integration-session side effect.

### 4.4 A range proxy is not the return variance — OPEN

HAR is fed the Parkinson proxy and its output becomes the variance of a
**close-to-close return** distribution. Those are different quantities: a
range proxy estimates intraday variance and excludes the overnight gap (~9% of
return variance even in the toy fixture, materially more on real equity
indices). So HAR's variance forecast is structurally biased low relative to
the target it is scored against, independent of any model error. This affects
QLIKE and every tail metric, and it applies to the real D-004 panel, where
`docs/research_design.md` specifies exactly this pairing (Parkinson + Garman–
Klass proxies for the indices).

### 4.5 One bad day can kill a whole cell — OPEN

Stream C's contract is that nothing is ever dropped: an unscorable origin
yields a row with NaN and a `missing_reason`. Stream B's GARCH honours the
spirit (falls back rather than raising), but **HAR raises** on non-positive or
non-finite RV, and `_run_block` has no per-origin exception guard. One
limit-locked day (high == low ⇒ Parkinson = 0 ⇒ `log(0)`) on a real index
therefore crashes the entire backtest for that cell, rather than costing one
row. The toy fixture cannot trigger it (prices carry four decimals and the
path is fine-grained); real rounded exchange data can.

### 4.6 Positional alignment is unguarded — MITIGATED

`run_backtest` aligns `series`, `proxy` and `fit_series` positionally and
validates only their *length*. Two same-length arrays off by one day would
score every forecast against the following day's realization — real leakage
that no existing test can see. `benchmarks/toy.py` now asserts the two series
still share a pandas index before converting to arrays (found by the
leakage-check audit), but nothing forces a caller to do the same.

---

## 5. What is still stubbed or absent

| Planned | Status |
|---|---|
| `DataAdapter` protocol | **not built** — three source-shaped module APIs |
| Machine-readable licence flags "enforced at packaging time" | **not built** — prose in `docs/data_licenses.md` only |
| Trading-calendar awareness in `TimeSeriesFrame` | **not built** — timestamps only, no session schedule |
| DM tests, MCS | **not built** — Phase 2 |
| FZ / FZ0 loss, Kupiec, Christoffersen | **not built** — hit indicators are recorded, tests are not |
| `Runner` (grid orchestration, GPU batching) | **not built** — `Executor` is the seam it plugs into |
| Process / Slurm executors | **not built** by design — serial only at M1 (D-011) |
| Models 6–13 (statsforecast, LightGBM, PatchTST, TSFMs) | **not built** — Phase 2/3 |
| Multi-horizon (h=5, 22) | **plumbed, unexercised** — `horizon` exists throughout; only h=1 run |
| Expanding-window splits | **not built** — rolling only |
| Crisis sub-sample tagging | **not built** |
| Real data in any benchmark | **blocked on licensing** (§2) |
| `SupportsUpdate` implementations | **not built** (§4.3) |

---

## 6. Three highest-risk items going into Phase 2

**Risk 1 — the refit schedule is not the protocol (§4.3).** The research
design's "refit every 21 days" currently means "freeze the forecast for 21
days" for every econometric baseline. Every headline number in the paper comes
off that schedule. If this is discovered late, every GARCH/HAR result must be
recomputed — and the whole Phase 3 grid with it. *Mitigation:* decide the
semantics now, implement `update()` on `FittedGARCH`/`FittedEWMA`/`FittedHAR`
behind a decision record, and add a test asserting the fit count and the
conditioning index separately. Cheap now, expensive after the grid runs.

**Risk 2 — scored quantities are not the quantities being forecast (§4.2,
§4.4).** Two independent versions of the same failure: the Student-t QLIKE
floor, and the range-proxy-vs-return-variance mismatch. Both produce
*plausible* numbers that are systematically wrong in a direction that
correlates with the thing under test — heavier tails, and the models fed range
proxies. This is the class of bug that survives to review and then invalidates
a result. *Mitigation:* make `forecast_moments` exact for heavy tails (a
parametric Student-t in `dist.py` is the cleaner fix), and settle explicitly
what HAR forecasts and what it is scored against, before Phase 2 adds nine
more models on top.

**Risk 3 — the D-004 data panel does not currently exist (§3, §2).** Stooq
forbids redistribution, gates automated download, and no longer serves SPX,
DJI or FTSE 100 at all — only unlicensed CFD proxies. Today the panel can only
be assembled by hand, one browser download at a time, and none of it can be
archived to Zenodo for reproducibility. Every hypothesis H1–H3 rests on that
panel. *Mitigation:* treat this as a sourcing decision needing a human call
(see below), not an engineering task; it will not resolve itself, and it
blocks the reproducibility story the CFP asks for as much as it blocks the
data.

---

## 7. Flagged for a human decision — not resolved here

1. **SPX / DJI / FTSE 100 provenance.** Accept Stooq's CFD proxies and
   document the substitution in the paper's data section, or source those three
   elsewhere (FRED, a licensed provider) and keep Stooq for the seven indices
   it still serves directly? Carried forward from stream A; unchanged.
2. **Binance redistribution of the derived RV series** — needed before the
   Zenodo-archival plan in `docs/data_sources.md` can proceed for the crypto
   arm.
3. **Refit semantics** (§4.3) — needs a decision record before implementation.
4. **`docs/decisions.md` has no entry for any of the above.** D-004 predates
   the finding that its named symbols no longer exist. A reopen of D-004 looks
   warranted; this session did not edit it, per CLAUDE.md.
5. **Doc drift.** `docs/design.md` was rewritten as-built (task 6) and the
   planning-folder copy is now behind it. `docs/research_design.md`,
   `docs/metrics_reference.md` and `docs/data_sources.md` were left untouched
   and are now out of date in specific places named above — the refit schedule,
   the index panel, and the "TO-CONFIRM" licence cells that stream A has since
   confirmed.
6. **`docs/decisionsV1.md`** is untracked in the working tree — an older copy
   of `docs/decisions.md` differing only in a redacted AAAI submission number.
   Left alone; delete it or commit it deliberately.
