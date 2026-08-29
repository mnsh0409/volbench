# Prompt L — Two measured defects: fix, re-run, re-certify

**Terminal:** the main integration checkout, on a **new branch off `feat/p3-analysis`**:

```bash
git switch feat/p3-analysis && git pull
git switch -c fix/p3-model-defects
```

**Session:** fresh and separately named — `claude -n vb-l-fixes`. Run `/effort high`.

---

## What this prompt is, and its one hard rule

Prompt K measured two defects. This prompt fixes exactly those two, re-runs exactly the affected cells, and re-certifies the evidence that the fixes invalidate. **This prompt changes numbers** — that is why it is on its own branch.

**Hard rule: change nothing else.** Not a hyperparameter, not a round cap, not a tolerance, not a model spec beyond what is named below. If you find something else that looks wrong, write it down and leave it. A second change riding along inside a numbers-changing commit is unattributable later.

Specifically forbidden, because it will be tempting: K found `lgbm` runs 100/100 boosting rounds at every origin, with train MSE improving 0.78 → 0.24 from 25 → 800 rounds. **Do not raise the round cap.** The out-of-fold fix already lands within 1.5% of the realized factor *at the current capacity*, so the smearing defect is fixed without touching capacity. Raising rounds would change the model rather than fix a bug. Record it as a known limitation and move on.

---

## Fix 1 — LightGBM: out-of-fold smearing

Current: the Duan factor comes from in-sample log residuals (`lgbm.py:390`), held across the refit block. Measured shipped 1.371 against a realized 1.678 — a ~22% understatement, same sign on all eleven assets, and **5.57 versus 1.56 inside COVID**, so the bias concentrates in the crisis sub-samples that are a headline result. The out-of-fold factor measured 1.703, within 1.5% of target.

1. Move the factor to out-of-fold residuals. K's probe (`benchmarks/lgbm_smearing_probe.py`) already establishes the construction and already carries a corrupt-the-future canary with a causal-boundary argument. **Port that canary into the shipped adapter's test suite** — a probe being leakage-clean is not evidence that the adapter is.
2. **Handle the cache trap deliberately.** K found the factor lives in `fit`, not `spec()`, so the config hash does not move — and `has()` short-circuits on file existence, so a naive re-run would report `cached 11, computed 0` and silently serve the old forecasts. Fix by **naming the smearing construction in `spec()`**, so the eleven hashes move for the right reason. Do **not** bump `package_version` — that would invalidate all 143 cells to fix eleven.
3. **Acceptance test:** after the re-run, recompute the realized-versus-shipped factor comparison and assert the shipped factor now matches realized within a stated tolerance. A fix without an assertion that it worked is a hope.

---

## Fix 2 — TSFM tail closure

Current: the grid's mean is computed with flat tails — 20% of the mass in two atoms at q₀.₁ and q₀.₉. K's diagnosis is that the mean is the right functional and the closure is the bug. Three closures give 1.09–1.30× the current variance; VaR/ES shift by exactly √ratio, +5.4% moirai to +9.6% timesfm.

1. **Adopt the lognormal closure**, unless you find a reason against it — and if you do, say what it is rather than substituting your own preference silently. The argument is not that it is the middle number: **realized volatility is approximately lognormally distributed**, which is one of the better-established stylized facts in the realized-volatility literature (Andersen, Bollerslev, Diebold & Labys). A lognormal tail closure on an RV quantile grid is therefore the choice with literature behind it. Cite it in the docstring so the choice is legible later.
2. **Keep the other two closures as a reported sensitivity**, not as dead code — a small committed function that reports all three on demand, so the paper can state the range. The tail beyond q₀.₁/q₀.₉ is genuinely unidentified; a single number would overstate what the data supports.
3. `variance_from` is already a `spec()` field, so the 33 hashes move on their own. Verify that rather than assuming it.

**Out of scope, deliberately:** the collapse of the RV distribution to a point variance is *not* being changed here. It is a disclosed design decision with K's measurements behind it, and a mixture return law may be added later as a separate arm. Do not implement one now.

---

## Re-run, and check the arithmetic

Re-run the grid. Then verify, and report, **exactly this**:

- **44 cells recomputed** — 11 `lgbm` (CPU) + 33 TSFM (GPU) — and **99 cells still cached, 0 failed.**
- If more than 44 recompute, a hash moved that should not have. Stop and report which, before doing anything else.
- If fewer than 44 recompute, a hash did not move that should have — the cache trap above. Stop and report.

---

## Re-certify what the fixes invalidated

**This is the step that gets forgotten.** J1's leakage canary certified the *old* `lgbm`, `chronos`, `timesfm` and `moirai` code paths. You have just replaced all four. The evidence has to follow the code.

1. Re-run `benchmarks/leakage_canary.py` on those four configs — all three legs (determinism, future corruption, past corruption), same as J1. Report the same two-line verdict per config. A "past-corruption: identical" is a stop-and-report.
2. Re-run the determinism gates: byte-identical serial versus parallel, and the resumability re-run showing every unchanged fragment untouched.
3. Full suite, `ruff`, `mypy --strict`, all three interpreters. Push the branch so CI sees it before any merge.

---

## Fold in while you are here

Three small things, all already agreed, none of which changes a number:

- **Commit the data digests.** `series_sha256` / `proxy_sha256` / `raw_sha256` currently exist only in gitignored sidecars, so a clean checkout can run the study but cannot verify it has the right inputs. Put a digest manifest under `docs/`.
- **Fold the per-asset aggregation into committed code** — K's tables came from scratchpad scripts, which makes them re-derivable but not re-runnable.
- **Record the interpreter in the run manifest.** J1 had to establish 3.11.5 three indirect ways because the run recorded it nowhere.

---

## Report

The 44/99/0 cell arithmetic; the acceptance test for Fix 1; the closure adopted and the three-closure sensitivity range; the four canary verdicts; the determinism gate results; CI status and the branch SHA. State plainly anything you changed that is not named in this prompt — there should be nothing.

**No interpretation of scores.** Do not compute or comment on whether any model now looks better or worse. That is J2's and J3's job, and the planning machine's after them.
