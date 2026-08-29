# Prompt O — Order statistics have already reached committed reports

**Branch:** stay on `fix/repo-hygiene` (or wherever N's work now lives; if it has been merged, branch `fix/order-statistics` from `feat/p3-analysis`).

**Session:** **N's session if it is still open** — it established the rule and knows which columns are which. Otherwise `claude -n vb-o-orderstats`, and read `docs/P3_REPO_HYGIENE.md` §3 first. Run `/effort high`.

**Size:** small. It regenerates reports; it re-fits nothing and touches no store.

---

## The finding

N refused to commit `docs/P3_CONVERGENCE_FITS.parquet` because `max_abs_return` is an order statistic — a single realised observation reproduced verbatim, and a sequence of them over overlapping windows discloses actual return magnitudes and brackets their dates. That reasoning is right.

**The same values are already committed in markdown**, under a different label. In `docs/P3_CONVERGENCE_FORENSICS.md`:

- **line 331** — `Every fit from 4993 onward has the same window maximum, |r| = 0.14183` — one exact HSI log-return magnitude, with its date bracketed by the range of origins over which it persists.
- **line 346** — a `window max |r|` row of quantiles *of maxima*: several of those numbers are themselves single observations, and `0.1418` recurs because it is one return seen in three windows.
- **line 350** — a per-origin table carrying both a `date` column and a `window max |r|` column. This is the worst of the three: roughly fourteen (magnitude, bracketed-date) pairs, which is precisely the disclosure the parquet was withheld to prevent.

Lines 398 and 404 in the same document are **already in the correct form** — `0 of 15 / 15 of 15`, and "larger at every one of the fifteen, by 1.106× to 1.801×". Counts and ratios. Use that section as the model for the rest.

Refusing the parquet while shipping the same values in prose is incoherent, and Stooq forbids redistribution with no de minimis clause.

## 1. Fix the values — ratios and ranks, not coarsening

Every argument these numbers support is **comparative**, so the comparison can be reported without the value:

- Line 331's point is that the maximum is *constant* from origin 4993 onward. Say that; drop the number. It adds nothing.
- Line 346: express each figure as a ratio to the asset's clean-group median, or as a rank.
- Line 350: either drop the column, or replace each entry with its ratio to that asset's clean median.

**Prefer ratios and ranks over coarsening.** Rounding `0.14183` to `0.142` still discloses three digits of a real observation, which is enough to match against the underlying series. A ratio discloses neither operand.

Regenerate from the parquet already on disk — do **not** re-fit the 7,101 models.

## 2. Sweep — this is not the only instance

Two starting points, then look wider:

- `docs/P3_CONVERGENCE_FORENSICS.md` tables T1–T8: check every one, not only the three lines above.
- **`docs/P3_ANALYSIS_VALIDITY.md`** — J1 reported "the minimum realized target" per asset, and that among the thirteen exactly-zero target days "the largest carries a **+4.13%** close-to-close return". Both are order statistics of a licensed-derived series. Same treatment.

Then check `P3_LOSS_TABLES`, `P3_PAIRWISE_COMPLETE`, `P3_MODEL_DEFECT_FIXES` and `P3_TSFM_VARIANCE_AUDIT`. My reading is that those carry means, standard errors, counts and ratios of *derived* quantities and are clean — but read them rather than trusting that.

The distinction to apply: **an aggregate over many observations is publishable; a function returning one realised observation is not.** A max, a min, "the largest day", a single quoted return or target value. A mean, a variance, a kurtosis, a count, a ratio of two of these — fine.

## 3. Make it structural, because the rule currently lives only in a document

Tag each derived column at construction in `benchmarks.convergence_forensics` as `aggregate` or `order_statistic`, and have the report writer **refuse to emit** anything tagged `order_statistic`. Apply the same policy to any artifact the module writes, so one declaration protects both the markdown and a future committed parquet.

Add the guard with its inert-proof, as M's and N's have: a fixture tagging a column `order_statistic` and asserting the writer refuses it. A policy nothing enforces is a memory, and this project has now been bitten six times by checks that could not distinguish "passed" from "did not run."

## 4. Verify and report

- The three lines above, and everything the sweep found, in their new form — quote the before and after.
- Every argument the document made before, it still makes.
- Guard passes; inert-proof fires.
- `git ls-files -- data/` empty; licensing guard green; `tests/test_identity_leakage.py` green.
- Full gate on all three interpreters; push, do not merge.

State plainly whether any other committed document carries an order statistic you did not fix, and why.
