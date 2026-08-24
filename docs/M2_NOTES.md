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
silent default — flagged for Phase 2.

## Reproduce

`make reproduce` rebuilds the fixture and the benchmark from scratch and is
byte-identical to the committed fragments. Old fragments remain in any
content-addressed `ResultsStore` under their M1 hashes; nothing is overwritten.
