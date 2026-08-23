# Phase 1 terminal prompts (M1 · due 27 Aug 2026)

Copy-paste prompts for Claude Code sessions in the volbench repo. Run **T0 alone first**, then A/B/C in parallel terminals, then **D** to integrate.

**Prerequisite:** run the setup script for your machine once — `SETUP.ps1` (Windows) or `SETUP.sh` (Linux). It builds the repo and the three worktrees, so T0 only has to verify them.

**How many terminals: 3 for the parallel phase** (T0 and D each run alone in one terminal, before and after). Two things force separate terminals rather than one session with subagents: you want a **different model per stream** (Sonnet for A/B, Opus/Fable for C — see Notes), and you want three independently resumable sessions you can watch and restart, instead of one orchestrator whose context fills with three streams' worth of detail and whose death loses all coordination.

**Why worktrees:** three Claude Code sessions cannot share one checkout — git allows only one branch checked out per working tree. T0 therefore creates a **separate git worktree per stream**, so each terminal gets its own directory on its own branch, sharing one object store. On top of that, each stream owns a **disjoint source directory** and is forbidden from touching the two shared choke points (`pyproject.toml`, `src/volbench/__init__.py`) — T0 settles those up front, D wires them at the end.

**Single-terminal variant:** run T0 → A → B → C → D sequentially in the main checkout, skipping every branch and worktree instruction. Slower in wall-clock (roughly the sum rather than the max) but simpler, and perfectly adequate against a 27 Aug M1.

**Two conventions fixed here so the streams cannot disagree** (they are in the prompts, repeated deliberately):

1. `FittedModel.predict(h)` returns a `Distribution` over the **next-period return**, not over variance. The variance forecast is a property of that distribution. QLIKE compares that variance against the proxy; CRPS / pinball / VaR / ES operate on the return distribution directly.
2. Variance is always in **daily units**, never annualized, anywhere in the library.

---

## T0 · Setup (run alone, first — ~10 min)

```
Read CLAUDE.md and docs/design.md first.

Task: prepare this repo for three parallel work streams. Do these in order and stop at the first thing that looks wrong rather than working around it.

1. Verify the bootstrap actually completed: `git log --oneline`, `git status`, `git remote -v`, and confirm `uv run pytest` passes (expect 51 tests), `uv run ruff check .` and `uv run mypy` are clean. If the repo is not in that state, STOP and report exactly what is missing — do not attempt repairs beyond obvious one-liners.

2. Add every Phase 1 dependency now, so no other stream ever edits pyproject.toml:
   `uv add pandas pyarrow requests arch statsmodels scipy`
   `uv add --dev pytest-cov`
   Re-run the three checks. Commit as `chore: add phase 1 dependencies`.

3. Create `docs/data_licenses.md` with an empty table (columns: source | series | frequency | license | redistributable | access method | checksum | verified_on) and a header note stating the rule: only redistributable sources may be vendored; everything else goes through a bring-your-own-data adapter. Commit.

4. Verify the worktrees SETUP.ps1 created: `git worktree list` should show data, models and eval on feat/data-layer, feat/model-adapters and feat/evaluation. If any is missing, create it (`git worktree add <path> <branch>`) and run `uv sync --dev` there.
   IMPORTANT: the dependencies you added in step 2 landed on main, so merge main into each stream branch and push, otherwise the three streams start without pandas/arch/etc.

5. Print a summary: current commit, test count, `git worktree list` output, and anything you flagged.

Do not implement data, model, or evaluation code in this session.
```

---

## A · Data layer — run in the `data` worktree

```
Read CLAUDE.md and docs/design.md first. You are in the `data` worktree, already checked out on branch feat/data-layer. Do not switch branches.

YOU OWN: src/volbench/data/**, tests/test_data_*.py, docs/data_licenses.md
DO NOT TOUCH: pyproject.toml, src/volbench/__init__.py, src/volbench/{dist,splitter,metrics}.py, src/volbench/models/**, src/volbench/{evaluate,results,execute}.py

Goal (M1 slice): a leakage-safe data layer that turns raw market data into daily variance targets.

Build:
1. `data/types.py` — TimeSeriesFrame: a frozen container holding a UTC DatetimeIndex, OHLC (or close-only) columns, an asset id, and a source tag. Validate: strictly increasing index, no duplicate timestamps, no NaN in required columns. Trading calendars differ per asset — never reindex two assets onto a shared calendar inside this layer.
2. `data/proxies.py` — daily variance proxies, all in DAILY units, never annualized:
   - squared return: r² where r = ln(C_t/C_{t-1})
   - Parkinson: (1/(4·ln2))·(ln(H/L))²
   - Garman–Klass: 0.5·(ln(H/L))² − (2·ln2 − 1)·(ln(C/O))²
   - realized variance from intraday bars: sum of squared intraday log returns within the day, with a `min_bars` guard that returns NaN for thin days
   Each takes explicit arrays/frames and returns a series; no hidden state.
3. `data/stooq.py` — downloader for the D-004 index set (^SPX ^NDX ^DJI ^DAX ^FTM/^FTSE ^CAC ^NKX ^HSI ^TWSE ^KOSPI — verify the exact Stooq symbols yourself and record them). Cache to a local parquet under `data/` (gitignored) keyed by symbol + download date, storing a SHA256 of the raw payload.
4. `data/crypto.py` — 1-minute bars for BTC-USD and ETH-USD from a public, permissively-licensed source; aggregate to daily realized variance at a configurable sampling interval (default 5 min).
5. Tests: proxy formulas against hand-computed values on tiny fixtures; property tests (all proxies non-negative; Parkinson ≈ squared-return in expectation on simulated GBM within tolerance); TimeSeriesFrame validation rejects unsorted/duplicate/NaN input; the RV aggregator returns NaN below min_bars. Use small committed fixtures — never hit the network in tests.

HARD RULES
- Before finishing, run the leakage-check skill over everything you wrote. Any rolling or aggregating function must use only data at or before its timestamp.
- Fill in docs/data_licenses.md for every source you actually implement, including the ToS URL and the date you checked it. If a source's terms do not clearly permit programmatic download and redistribution of derived series, implement it as bring-your-own-data and say so — do NOT vendor data of uncertain provenance.
- Never commit downloaded data.
- `uv run pytest && uv run ruff check . && uv run mypy` must be green before every commit. Commit incrementally with conventional-commit messages.
- Push the branch. Do NOT merge to main and do NOT open a PR.
- Stop and report if a data source is unreachable, its licence is ambiguous, or a decision would contradict docs/research_design.md + docs/decisions.md (D-004).
```

---

## B · Model adapters — run in the `models` worktree

```
Read CLAUDE.md and docs/design.md first. You are in the `models` worktree, already checked out on branch feat/model-adapters. Do not switch branches.

YOU OWN: src/volbench/models/**, tests/test_models_*.py
DO NOT TOUCH: pyproject.toml, src/volbench/__init__.py, src/volbench/{dist,splitter,metrics}.py, src/volbench/data/**, src/volbench/{evaluate,results,execute}.py

CONVENTION (fixed — do not redesign): `FittedModel.predict(h)` returns a `Distribution` over the NEXT-PERIOD RETURN, not over variance. A model's variance forecast is a property of that distribution. All variance is in DAILY units, never annualized.

Goal (M1 slice): the baseline model family, behind one adapter protocol.

Build:
1. `models/base.py` — `ForecastModel` protocol: `fit(train: np.ndarray, **ctx) -> FittedModel`; `FittedModel.predict(h: int) -> Distribution`. Include a `name` property and a `spec()` returning a JSON-serializable dict of hyperparameters (the evaluation stream hashes this — keep it stable and sorted).
2. `models/naive.py` — random-walk volatility: next-period sigma = trailing realized sigma over the fit window.
3. `models/ewma.py` — RiskMetrics EWMA, default lambda 0.94, recursion sigma²_t = λ·sigma²_{t-1} + (1−λ)·r²_{t-1}; expose lambda in spec().
4. `models/garch.py` — GARCH(1,1) and GJR-GARCH via the `arch` package, normal and Student-t innovations. Return a Normal for normal innovations; for Student-t, return a quantile-grid or sample-based Distribution (dist.py has constructors for both — pick one and document why). Guard against non-convergence: catch, log, and fall back to the EWMA forecast for that origin rather than raising, and record the fallback in the fitted object.
5. `models/har.py` — HAR-RV (Corsi 2009): OLS of RV_{t+1} on [1, RV_d, RV_w(5), RV_m(22)], fit in logs with a documented retransformation. Its `fit` takes a realized-variance series; state the input contract clearly in the docstring since it differs from the return-based models.
6. Tests: each model fits and predicts on simulated data; a GARCH fit on data simulated from known parameters recovers them within tolerance; EWMA matches a hand-computed recursion on a 5-point series; HAR recovers known coefficients on synthetic data built from its own design matrix; every model returns a Distribution with positive variance; spec() is stable across identical constructions and differs across different ones; the GARCH non-convergence path returns a usable forecast.

HARD RULES
- Run the leakage-check skill before finishing: nothing in fit() may see data past the fit window, and no transform may be fitted on the full series.
- Models must never import from volbench.data or volbench.evaluate — they take plain arrays. This keeps the adapter contract clean.
- `uv run pytest && uv run ruff check . && uv run mypy` green before every commit; conventional commits; push the branch; do NOT merge or open a PR.
- Stop and report if the return-distribution convention above cannot be honoured for some model — that is a design question, not something to work around.
```

---

## C · Evaluation & results — run in the `eval` worktree

```
Read CLAUDE.md and docs/design.md first. You are in the `eval` worktree, already checked out on branch feat/evaluation. Do not switch branches.

YOU OWN: src/volbench/evaluate.py, src/volbench/results.py, src/volbench/execute.py, tests/test_evaluate.py, tests/test_results.py, tests/test_execute.py
DO NOT TOUCH: pyproject.toml, src/volbench/__init__.py, src/volbench/{dist,splitter,metrics}.py, src/volbench/data/**, src/volbench/models/**

CONVENTION (fixed): models return a `Distribution` over the next-period RETURN; its variance is the variance forecast. QLIKE compares that variance to a proxy; CRPS/pinball operate on the return distribution. Daily units throughout. Define a small structural Protocol locally for the model interface rather than importing volbench.models — that stream is being built in parallel.

Goal (M1 slice): run a model over a series through RollingOriginSplitter and store scored, reproducible results.

Build:
1. `results.py` — `config_hash(spec: dict) -> str`: stable SHA256 over a canonically-serialized dict (sorted keys, fixed float formatting) of {model spec, data spec, splitter params, seed, package version}. `ResultsStore`: append-only parquet, one row per (config_hash, asset, origin_index, horizon) with the scores, the forecast's mean and variance, the realized target, and the proxy used. Idempotent: re-running an identical config must not duplicate rows, and a cache lookup by config_hash must short-circuit the whole run.
2. `evaluate.py` — `run_backtest(model_factory, series, proxy, splitter, seed, ...)`: iterate `splitter.split(n)`, refit only when `origin.refit` is True (reuse the previous fit otherwise), predict, score, and return a tidy frame. Scores per origin: CRPS, log score where defined, pinball at {0.01, 0.025, 0.05}, QLIKE of forecast variance vs proxy, and the VaR hit indicator at each level. Handle NaN proxies by scoring what is possible and recording why the rest is missing — never silently drop rows.
3. `execute.py` — an execution seam, implemented SERIAL only for now: an `Executor` protocol with `map(fn, items) -> list` and a `SerialExecutor`. Every backtest must route its per-cell work through an Executor rather than looping directly, so Phase 2 can add local-multiprocessing and Slurm-array backends without touching evaluation logic. Do NOT implement those backends now — just make the seam exist and be used, and note in the docstring that a cell is a (asset, model, splitter, seed) unit whose results merge by config_hash.
4. Tests: a perfectly-specified model beats a deliberately misspecified one on CRPS and QLIKE; config_hash is stable across runs and dict orderings, and changes when any component changes; the store is idempotent under re-run and the cache short-circuits; refit cadence is honoured (count actual fits with a spy model and compare against the expected count from refit_every); an end-to-end run on simulated data is bit-identical across two runs with the same seed.
5. A determinism test that is the repo's canary: two full runs with identical seed produce identical parquet contents.

HARD RULES
- Run the leakage-check skill before finishing. Every train/test index MUST come from RollingOriginSplitter — no hand-rolled slicing anywhere, including in tests.
- `uv run pytest && uv run ruff check . && uv run mypy` green before every commit; conventional commits; push the branch; do NOT merge or open a PR.
- Stop and report if scoring a model cleanly requires changing dist.py, splitter.py, or metrics.py — those are shared and frozen for this phase.
```

---

## D · Integration & M1 (after A, B, C are pushed)

```
Read CLAUDE.md and docs/design.md first.

Task: merge the three Phase 1 streams and close milestone M1.

1. From the MAIN checkout (not a stream worktree), merge feat/data-layer, feat/model-adapters, feat/evaluation into main in that order, resolving conflicts conservatively. If any branch is not green on its own, fix it in its own worktree first. After a clean merge and tag, remove the three worktrees with `git worktree remove`.
2. Wire src/volbench/__init__.py: export the new public surface, keep __all__ sorted.
3. Reconcile the model interface: stream C defined a local Protocol, stream B the real classes. Make them agree — adjust C's Protocol to B's reality unless B violates the return-distribution convention, in which case fix B.
4. Write the M1 end-to-end smoke test: one real asset (or a committed fixture if the network is unavailable), naive + EWMA + GARCH(1,1) + HAR, ~200 rolling origins, producing a scored results table. It must run in under two minutes and be deterministic under a fixed seed.
5. Make `make reproduce` actually rebuild that toy benchmark from scratch.
6. Update docs/design.md to match what was built — the as-built API, not the plan. Note every place they diverged.
7. Run the full check suite, confirm CI is green on GitHub, tag `v0.1.0-m1`, push tags.
8. Write docs/M1_REPORT.md: what exists, what each stream deviated on, what is still stubbed, measured runtime for the toy benchmark, and the three highest-risk items going into Phase 2.

Stop and report rather than papering over any conflict between the streams' assumptions — those disagreements are exactly what this session exists to surface.
```

---

## Notes

- **Watch the first 10 minutes of each stream.** Autonomous runs go wrong early or not at all; once a stream has committed twice cleanly it usually stays on the rails.
- **Model choice per D-006 economics:** Sonnet 5 at high effort is right for A and B (mechanical, well-specified). Use Opus or Fable at xhigh for C and D — C carries the determinism and caching logic, D resolves cross-stream disagreements; both are where a subtle wrong answer costs days.
- **If a stream finishes early,** the next work is Phase 2 model adapters (statsforecast, LightGBM) for B, and the parallel runner for C — but merge M1 first.
