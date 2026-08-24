# M2 notes — evaluator hardening (branch `m2/evaluator-hardening`)

Post-M1 fixes to the four open items in `docs/M1_REPORT.md` §4. This file
records the target-estimator change (§4.4) and its measured effect on the toy
benchmark, including where the effect runs counter to expectation — the toy is
a smoke signal, not evidence, and it is reported faithfully.

## What changed (§4.4)

HAR's scoring target is now a per-day **close-to-close** variance estimator,
`overnight_plus_range_variance` = `(ln(O_t/C_{t-1}))² + RS_t`, with `RS_t` the
Rogers & Satchell (1991) drift-independent range term. It is NOT Yang-Zhang:
YZ is a windowed multi-day estimator, and a window reaching past day `t` would
put the future into day `t`'s target. Return-fed models (naive, EWMA, GARCH,
GARCH-t) keep Parkinson. A Student-t GARCH config was added so `make reproduce`
exercises the parametric `StudentT` path (D-014), not only the unit tests.

The toy fixture was regenerated: its overnight jump and intraday path are now
**independent** components whose variances sum to a *recorded* true daily
variance (the M1 generator carved the gap out of the return, which coupled the
pieces and left the intraday variance at 1.09× the truth). This makes the
estimator bias measurable against a known target, and it moved every number in
the benchmark. Intraday resolution went 390→5000 steps so range-estimator
discretization (~8% at 390) no longer masquerades as the ~9% overnight effect.

## The estimator is validated against the known truth

`tests/test_target_estimators.py`, 20 000 days, discretization negligible:

| estimator | bias vs true close-to-close var | sampling variance of ratio |
|---|--:|--:|
| `overnight_plus_range` | **−2.9%** | **0.28** |
| Parkinson | −11.0% (≈ the 9% overnight gap) | 0.33 |
| Garman-Klass | −11.7% | 0.21 |
| squared return | +0.7% (unbiased) | 2.01 (noisy) |

So the new target's bias is ~a quarter of Parkinson's (it captures the
overnight jump Parkinson structurally omits) and its sampling variance is ~7×
below the squared return's — the two claims the decision rests on. Rogers-
Satchell's drift-independence is pinned separately: under a drift of 2× the
daily vol, RS moves <5% while Parkinson inflates >100%.

## Effect on the toy benchmark — decomposed

Because the fixture changed, every model's numbers move. To separate the two
causes, three runs: **OLD** (M1 fixture, all Parkinson) → **MID** (M2 fixture,
all Parkinson) → **NEW** (M2 fixture, HAR on the new target).

`OLD→MID` is the fixture change; `MID→NEW` is the target change. The target
change touches **HAR only** — every return-fed model is byte-identical between
MID and NEW (each model is an independent cell; verified exactly in
`tests/test_toy_targets.py`). QLIKE (vs the model's own proxy):

| model | OLD | MID | NEW | fixture Δ | target Δ |
|---|--:|--:|--:|--:|--:|
| naive | 0.3300 | 0.3654 | 0.3654 | +0.0355 | **0** |
| ewma | 0.2035 | 0.1747 | 0.1747 | −0.0288 | **0** |
| garch11 | 0.2177 | 0.2152 | 0.2152 | −0.0026 | **0** |
| garch11_t | n/a | 0.2152 | 0.2152 | — | **0** |
| har | 0.2155 | 0.1781 | 0.1823 | −0.0375 | +0.0043 |

## Where the toy contradicts the expectation — reported, not papered over

The expectation was that HAR's QLIKE and tail hits would *improve* under the
correct target. **On this fixture they do not.** HAR's QLIKE-vs-its-proxy rises
slightly (0.178→0.182), and the honest, sample-size-robust diagnostic — HAR's
forecast variance against the fixture's *known* true close-to-close variance —
gets worse:

| HAR fed / scored on | forecast_var / true | QLIKE vs **true** | VaR hit @5% |
|---|--:|--:|--:|
| Parkinson (M1 setup) | 0.98 | 0.0077 | 0.045 |
| overnight_plus_range (M2) | 1.13 | 0.0263 | 0.035 |

Two reasons, both real:

1. **Parkinson-fed HAR was accidentally well-calibrated on this fixture.**
   Parkinson is ~9% low, but HAR's lognormal retransformation
   `E[RV]=exp(ŷ+½·resid_var)` pushes the level back up, landing at 0.98× the
   true variance. That cancellation is a property of this fixture's overnight
   share and noise, not a general one — on a real index with a larger, more
   persistent overnight component the intraday-only proxy would leave HAR
   biased low, which is the §4.4 concern.
2. **The correct target is noisier in log-space** (the overnight jump is a
   near-chi-squared single shock), so HAR's retransformation inflates more and
   the forecast overshoots to 1.13× true. HAR's Gaussian-log-residual
   assumption is mis-specified for a target with a heavy overnight tail.

The n=200 VaR hit rates (1–2 exceedances expected at 1%) are within sampling
noise and prove nothing either way at this sample size.

**Conclusion.** The estimator change is correct on principle — scoring a
close-to-close variance forecast against a close-to-close proxy is right, and
the estimator is validated against the truth. But the toy cannot demonstrate a
HAR *improvement*, and it surfaces a genuine, separate issue: **HAR's lognormal
retransformation is sensitive to the target's log-space noise.** That is a
Phase-2 modelling item (a bias-corrected or component overnight+intraday HAR),
now recorded here and in `docs/design.md`.

A second tension worth flagging: the return-fed models still score QLIKE
against **Parkinson** (intraday), yet they too forecast the close-to-close
variance — so their QLIKE remains mismatched in the same way HAR's return-based
scores were at M1. Fixing only HAR's target, per the M2 scope, is therefore a
*partial* correction. Scoring every model's QLIKE against the close-to-close
proxy is the consistent end state and a decision to take deliberately, not a
silent default — flagged for Phase 2. *(Taken: see "One scoring target per
cell" below — the per-model wiring lasted one commit.)*

## One scoring target per cell (M2 review, item 1 — supersedes the per-model wiring)

Principle: **the scoring target is a property of the evaluation cell, never of
the model.** With per-model targets, the QLIKE column compared models against
different proxies — which is not a comparison. Now every cell in a run scores
QLIKE against `overnight_plus_range` (all models forecast the close-to-close
variance, so all are scored against its estimator); HAR's *fit input* stays the
overnight-plus-range series under any setting, because what a model forecasts
is a modelling contract, not an evaluation knob. Parkinson survives as a
**labeled robustness arm** behind `--target parkinson` / `run_toy_benchmark(
target=...)` — one target per run, in the config hash, never a silent default.

The now-comparable table (`make reproduce`, 200 origins, refit every origin):

| model | CRPS | log score | QLIKE vs OPR | hit@1% | hit@5% | QLIKE before (per-model targets) |
|---|--:|--:|--:|--:|--:|--:|
| ewma | 0.00582 | −3.186 | **0.162** | 0.005 | 0.040 | 0.175 (Parkinson) |
| har | 0.00586 | −3.160 | **0.182** | 0.005 | 0.035 | 0.182 (already OPR) |
| garch11_t | 0.00587 | −3.160 | **0.173** | 0.005 | 0.035 | 0.215 (Parkinson) |
| garch11 | 0.00587 | −3.160 | **0.173** | 0.005 | 0.035 | 0.215 (Parkinson) |
| naive | 0.00601 | −3.078 | **0.303** | 0.010 | 0.030 | 0.365 (Parkinson) |

**The return-fed models' QLIKE levels shifted because the target changed, not
the models.** Their forecasts are untouched: CRPS, log score, pinball and hit
columns are byte-identical between the two wirings (the proxy never reaches a
model — pinned in `tests/test_toy_targets.py`), and only the QLIKE and proxy
columns moved. HAR's cell is literally the same experiment as before (same
config hash). With one target the QLIKE ranking is now meaningful: EWMA and
the GARCHes score better against the close-to-close proxy than against the
intraday one they were previously mismatched with, and naive remains worst.

## Version 0.2.0 — every config hash changed at this boundary, intentionally

`package_version` is part of every config hash. The M2 behaviour changes
(StudentT forecasts, refit semantics, the shared close-to-close target)
changed what a given config *computes*, but under 0.1.0 its hash — and so any
`ResultsStore` entry made by the old code — stayed servable. The bump to
0.2.0 closes that stale-hash hazard (flagged at the StudentT commit): every
hash moves, all pre-0.2.0 fragments are orphaned (never overwritten, never
served), and `make reproduce`'s pinned identities were regenerated once, here.

## Reproduce

`make reproduce` rebuilds the fixture and the benchmark from scratch and is
byte-identical to the committed fragments. Old fragments remain in any
content-addressed `ResultsStore` under their M1 hashes; nothing is overwritten.
