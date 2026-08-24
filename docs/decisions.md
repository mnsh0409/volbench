# 09 · Decisions log (append-only, newest on top)

> One entry per settled decision. Claude must not relitigate SETTLED entries unless explicitly reopened. PENDING/PROPOSED entries are open work.

## Format

```
## D-0xx · YYYY-MM-DD · Title            [SETTLED | PROPOSED | PENDING | REOPENED]
Decision: …
Why: …
Alternatives rejected: …
Revisit-if: …
```

---

> NUMBERING NOTE (added by the m2/evaluator-hardening session, 2026-08-24):
> D-013..D-016 below were appended from Claude Code to mirror decisions taken
> during the M1 integration and M2 hardening work. D-014 matches the number
> the task used for the QLIKE fix; the others are placed adjacent in the order
> the decisions were listed. **D-012 is not mirrored here** — it belongs to the
> planning machine, which is the source of truth for numbering. Reconcile
> against the planning log and renumber if it disagrees.

## D-016 · 2026-08-24 · HAR scoring target = per-day overnight + Rogers-Satchell   [SETTLED]
Decision: HAR-RV is fed and scored against a per-day CLOSE-TO-CLOSE variance estimator, `overnight_plus_range_variance` = `(ln(O_t/C_{t-1}))^2 + RS_t` with `RS_t` the Rogers & Satchell (1991) drift-independent range term — not plain Parkinson, and not literal Yang-Zhang.
Why: HAR forecasts the variance of the next close-to-close return, so it must be scored against an estimate of that quantity. A range proxy (Parkinson/Garman-Klass/RS alone) estimates only the intraday open-to-close variance and structurally omits the overnight jump (~9% of return variance in the toy fixture, more on real indices), biasing the scored forecast low (M1 report §4.4). Validated against the toy generator's known true variance: the new target's bias is ~a quarter of Parkinson's and its sampling variance ~7x below the squared return's (`tests/test_target_estimators.py`). RS is used, not Parkinson, for the intraday piece because it stays unbiased under drift; formulas corroborated 2026-08-24 across CRAN TTR, arXiv:1803.07152 and portfoliooptimizer.io (primary papers paywalled).
Alternatives rejected: (a) Yang-Zhang — a windowed multi-day estimator; per-day it is undefined, and a window reaching past day t would put the future into day t's target (look-ahead). (b) squared daily return — unbiased but ~7x noisier. (c) leaving HAR on Parkinson — the M1 mismatch.
Revisit-if: two open consequences, both in docs/M2_NOTES.md. (1) HAR's lognormal retransformation is sensitive to the target's log-space noise — on the toy this makes the "correct" target slightly worse-calibrated to the truth than the accidentally-well-calibrated Parkinson-fed HAR; a bias-corrected or component overnight+intraday HAR is the Phase-2 fix. (2) The return-fed models still score QLIKE against Parkinson though they too forecast close-to-close variance; scoring every model against the close-to-close proxy is the consistent end state, to be decided deliberately.

## D-015 · 2026-08-24 · Refit protocol = re-estimate every N, re-condition daily   [SETTLED]
Decision: "refit every N days" (D-014-planning: N=21) means parameters are re-estimated every N origins AND the model's conditional state is re-filtered on every origin's window in between (`recondition="daily"`, the default). The frozen behaviour — the forecast issued at the refit origin held until the next — stays available as an explicit ablation arm (`recondition="none"`), never as a default.
Why: at M1 no model implemented re-conditioning, so `refit_every=21` silently meant "freeze the forecast for 21 days", ignoring three weeks of realized returns — not the protocol docs/research_design.md describes, and a misleading reported cadence (M1 report §4.3). `update()` re-filters at fixed parameters (GARCH via arch's `ARCHModel.fix`, verified to reproduce the fit's conditional variances to <1e-8; EWMA/HAR/naive by their own recursions), never re-estimating.
Alternatives rejected: refit-only-no-recondition as the default (the M1 behaviour — understates the information available at each origin); off-schedule refit on the fly (would change the cadence the config hash records).
Revisit-if: per-model refit-schedule overrides are still open. `recondition` enters the config hash only when `refit_every>1` (it cannot change a number otherwise).

## D-014 · 2026-08-24 · QLIKE bias fix = parametric StudentT distribution   [SETTLED]
Decision: Student-t GARCH forecasts return a parametric `StudentT(loc, scale, df)` (closed-form mean/variance/CRPS) instead of a 199-point quantile grid; `forecast_moments` reads a distribution's own closed-form moments before falling back to a grid estimate.
Why: the grid spanned tau in [0.005, 0.995] with flat tails, and the evaluator's moments of that grid truncated the Student-t's tails — understating the variance ~24% at nu=3, so a *perfectly specified* forecast scored a QLIKE floor of 0.0407 instead of 0 (M1 report §4.2), growing exactly where the Student-t spec is meant to win. The parametric object removes the bias at its source with no RNG, so scoring stays bit-identical across runs. CRPS verified against numerical integration of two independent representations.
Alternatives rejected: tail-extrapolating the grid's moments (a patch on a lossy representation); sample-based Student-t (needs an RNG, breaks determinism).
Revisit-if: n/a — closed at the root.

## D-013 · 2026-08-23 · Milestone M1 complete; Phase 1 design confirmed   [SETTLED]
Decision: M1 (leakage-safe evaluation skeleton) is complete and tagged `v0.1.0-m1`. The Phase 1 design — RollingOriginSplitter as the sole index source, Distribution as the only forecast currency, content-addressed ResultsStore, serial Executor seam — held under integration and is confirmed as the base for Phase 2.
Why: three parallel streams (data, models, evaluation) merged with zero conflicts on disjoint file ownership; a toy benchmark runs four baselines over 200 rolling origins deterministically in ~2s; `make reproduce` rebuilds it from scratch. Full record in docs/M1_REPORT.md.
Alternatives rejected: n/a — milestone confirmation.
Revisit-if: the four §4 open items are addressed on m2/evaluator-hardening (D-014, D-015, D-016 close §4.2, §4.3, §4.4; §4.5/§4.6 fixed earlier on the branch). Highest remaining Phase-2 risks are recorded in docs/M1_REPORT.md §6.

## D-011 · 2026-08-22 · Slurm A100 cluster = scale backend, not dev machine   [SETTLED]
Decision: keep interactive development on the 4090 (D-010); use the Slurm A100 cluster as an EXECUTION BACKEND for the Phase 3 grid and ablations. volbench gets a pluggable `Executor` seam from Phase 1 (serial now; local-multiprocessing and Slurm-array backends in Phase 2), with results merged by config_hash in the ResultsStore.
Why not develop on Slurm: batch queues make iteration latency unpredictable; Claude Code wants a persistent interactive shell; login nodes are not for compute. Why not skip Slurm: it converts CFP pillar 3 (forecasting at scale) from a paragraph into a measured result.
PAPER OPPORTUNITY (the real reason this matters): §5 can report one scaling curve — single-core → 4090 multiprocess → Slurm array — on identical work, with config-hash provenance proving the results are bit-identical across backends. That is exactly the "efficient code, parallelisation" pillar, and almost no benchmark paper demonstrates backend-invariance of its own numbers. Upgrade H4 accordingly: same forecasts, three execution paths, measured speedup and verified identity.
Cluster gotchas to check BEFORE Phase 3 (each has bitten this kind of run before):
- do compute nodes have internet? If not, pre-stage all market data AND the TSFM weights (HF cache) on the shared filesystem from the login node;
- storage quota vs. the parquet ResultsStore and model weights;
- CUDA/module system vs. uv-managed venvs on a shared FS;
- queue policy and typical wait — never put a deadline-critical run behind an unknown queue.
Note on per-device speed: for 200–500M-parameter TSFMs an A100's advantage over a 4090 is modest (higher HBM bandwidth, comparable bf16 tensor throughput); the cluster's value is CONCURRENCY across many cells, not single-cell speed. Phase 3's econometric refits are the CPU-heavy, embarrassingly parallel part and benefit most.
Revisit-if: queue waits exceed ~a day, or cluster policy forbids long array jobs — then the 4090 alone still carries Phase 3, just with fewer ablation configurations.

## D-010 · 2026-08-22 · Machine roles: 4090 (Ubuntu) is the code machine   [SETTLED]
Decision: the 4090 Ubuntu box running Claude Code is the primary DEVELOPMENT machine for volbench; the 3090 Windows box running Cowork stays the PLANNING machine (knowledge files, prompts, correspondence, decisions). GitHub is the only seam between them. Repo canonical home: `~/volbench` on the 4090, worktrees `~/volbench-wt/{data,models,eval}`.
Why (revises the earlier 3090-primary call, made before the 4090's OS was known):
- Linux uses fork-based multiprocessing; Windows uses spawn. The Phase 2/3 parallel runner over (asset × model × origin) is exactly the workload where that gap bites — spawn re-imports per worker and is markedly slower and more fragile.
- The GPU and the code then live on one box, so Phase 3 TSFM inference needs no migration mid-schedule; the machine also already carries a proven CUDA/TensorRT/torch stack from the PVN work (D-009).
- Toolchain friction is lower (no SmartScreen/AV prompts on freshly built venv executables, cleaner `arch`/scipy builds).
Cost accepted: Cowork cannot bridge to the 4090 (no desktop app on it, and no IP/SSH path exists), so Claude in *this* kind of session is blind to the repo. Mitigation: terminal prompts and technical knowledge mirrors ship inside the repo under `docs/`, so they travel by git; masters stay on the 3090 where Cowork maintains them. READ PATH (added 22 Aug): the repo is public, so Cowork sessions can fetch any pushed file via `https://raw.githubusercontent.com/<user>/volbench/<branch>/<path>`. Working habit therefore: **push before asking Claude to review**. Write access to the 4090 remains manual (paste, or an optional Samba/SSHFS mount connected as a Cowork folder — untested).
Setup order: `SETUP.ps1` on Windows once (seeds and pushes to GitHub) → `SETUP.sh` on the 4090 (clones, syncs, builds worktrees). SETUP.sh requires the GitHub repo to exist.
Revisit-if: a Claude desktop app becomes available for the Ubuntu box, or the 4090 becomes unavailable — the Windows path remains fully working as a fallback.

## D-009 · 2026-08-22 · AAAI/Go (PVN) project stays out of the IJF SI   [SETTLED]
Decision: do NOT redirect the Pedagogical Value Networks (KataGo/Go) work to the IJF Open-Source Forecasting SI. volbench remains the IJF submission.
Why: (1) BLOCKING — PVN is an anonymous submission under review at AAAI 2027; its own cover page forbids distribution, and concurrent submission violates both venues' rules, exactly as for TCSS (D-007). (2) Fit — the contribution is a search-selection rule (P-MCTS) plus a soundness theorem, i.e. game-AI/RL theory; the CFP explicitly excludes work that is not about how forecasting problems shaped forecasting *software* design. (3) Speed — nothing in it is closer to an IJF manuscript than volbench already is, so switching costs weeks rather than saving them.
Reusable anyway: the paper's forecast-evaluation craft transfers to volbench §5 — decile calibration plots against the perfect-calibration diagonal; the within-position vs pooled correlation decomposition (adopt this whenever reporting correlations — pooled figures are inflated by between-unit variance); pre-registered kill criteria stated up front; per-result wall-clock and artefact-to-script maps.
Capacity risk logged: three projects live, two in review (TCSS reviews ~30 Oct; AAAI reviews expected Oct–Nov). Both revision windows overlap volbench Phases 3–4. If both land together, volbench *writing* slips before volbench *rigor* does.
Hardware note: the PVN paper reports runs on one RTX 4090 with KataGo/TensorRT, and master_codebase ships working `katago_trt_env.sh` / `torch_cudnn_env.sh` — so that box already has a proven CUDA/TensorRT/torch stack, strengthening the "4090 as Phase 3 TSFM compute node" plan.
Revisit-if: AAAI rejects AND a genuinely forecasting-shaped paper is carved out (e.g. open-source calibration/evaluation tooling for value-network win-probability forecasts) — a new paper for a later cycle, after volbench v1 ships.

## D-008 · 2026-08-19 · Accelerated timeline v2 (full-time mode)   [SETTLED — reviewed with Martin]
Decision: supersede 00_plan §5 dates. Labor-bound phases compress ~7× per D-006; calendar compresses ~3× because compute wall-clock (grid sweeps under the D-005 GPU budget), verification attention (all agent-written temporal logic needs manual sign-off — leakage bugs are silent), and third-party latency (editor reply, external readers) do not scale with hours.
- Phase 1 design & data: 20–27 Aug · M1 (end-to-end toy benchmark, CI green)
- Phase 2 core build: 27 Aug → 12 Sep · M2 (full grid runs, 12 assets × 13 models)
- Phase 3 experiments & robustness: 12–30 Sep · M3 (results frozen, `make reproduce`) — compute-bound
- Phase 4 writing: 20 Sep → 12 Oct (overlapped) · M4 (complete draft + repro package)
- Phase 5 submit: **1–15 Oct 2026** (floor: 30 Sep only if editors grant exactly that window)
Optional accelerators (not adopted): raise D-005 to ~$500 spot-GPU (≈1 week off Phase 3); skip external readers (≈10 days, higher rejection risk — keep readers).
Hard guard: the robustness/ablation block is never compressed to hit a date; if a granted extension is shorter than the floor, take the regular track. Pre-verify data licenses in week 1 so the pipeline never stalls. TCSS reviews (~30 Oct) land post-submission in this schedule.

## D-007 · 2026-08-19 · TCSS/IJF relationship         [SETTLED]
Decision: keep tracks separate — do not rework the in-review TCSS manuscript for IJF; execute M1–M6 from 01_planning/TCSS_vs_IJF_comparison.md. Per the current Guide for Authors, the IJF cover letter MUST disclose the TCSS manuscript as similar-methods work under review and explain the difference (different task, data, contribution).
Why: concurrent-submission rules; poor IJF-CFP fit as an application paper; two-paper portfolio is stronger.
Revisit-if: TCSS rejects — reframed ideas (not text) could seed a later software paper after volbench v1.

## D-006 · 2026-08-19 · Time budget                   [SETTLED]
Decision: FULL-TIME. 12+ h/day manual work + ~24 h/day automated (Claude agents, batch runs) until a financial-software-engineering role starts; then drop to evenings and revert toward the original cadence.
Why: Martin's stated capacity. Drives D-008 accelerated timeline. Sunday P01 review stays (30 min).

## D-005 · 2026-08-19 · Compute budget                [SETTLED]
Decision: ≤ US$200 total. Kaggle free GPU quota first, Colab Pro+ only in heavy months, spot GPU only if the Phase 3 grid demands it. TSFMs zero-shot only (no fine-tuning). Aggressive result caching keyed by config hash.

## D-004 · 2026-08-19 · Core data panel               [SETTLED]
Decision: 10 equity indices via Stooq daily OHLC (S&P 500, NASDAQ-100, Dow, DAX, FTSE 100, CAC 40, Nikkei 225, Hang Seng, TAIEX, KOSPI) + BTC-USD, ETH-USD 1-min exchange data → daily RV. Span 2005-01 → freeze date (crypto from listing). Proxy hierarchy: crypto 5-min RV; indices Parkinson + Garman–Klass range proxies + squared-return robustness. Crisis sub-samples: GFC Sep 08–Mar 09, COVID Feb–Apr 20, 2022 tightening, Aug-2024 spike, latest 2025–26 stress window (fixed at grid freeze).
Depends on: 06_data_sources license confirmation for Stooq + chosen exchange (verify before shipping data).

## D-003 · 2026-08-19 · License                       [SETTLED]
Decision: Apache-2.0.
Why: patent grant; matches wrapped ecosystem (statsforecast, GluonTS); corporate-friendly.
Alternatives rejected: MIT (no patent grant).

## D-002 · 2026-08-19 · Package name                  [SETTLED]
Decision: **volbench**.
Evidence (18 Aug 2026): pypi.org/project/volbench → 404 (free); no exact GitHub match (nearest: VBench video benchmark, wesm/vbench, VulBench — different domains).
Action NOW: claim the PyPI name with a 0.0.1 placeholder + create the GitHub repo today (Phase C step 11).

## D-001 · 2026-08-18 · Project scope & angle         [SETTLED]
Decision: Open-source framework for reproducible evaluation of probabilistic volatility & tail-risk forecasts (GARCH→TSFMs), targeting IJF SI "Open-Source Forecasting"; two-track deadline strategy per 00_plan.
Why: fits CFP pillar 3 (+1,2,4); plays to engineering strengths; fills finance-specific gap; portfolio value.
Alternatives rejected: new general forecasting library (crowded); production case study (no operational system to report); pure upstream contribution (slow to paper, kept as ~10% side track).
Revisit-if: guest editors reject scope, or a directly overlapping finance benchmark ships before October 2026.
