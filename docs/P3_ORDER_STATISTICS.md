# P3 — order statistics had already reached committed reports

docs/P3_REPO_HYGIENE.md §3 refused to commit `docs/P3_CONVERGENCE_FITS.parquet`
because one of its 49 columns, `max_abs_return`, is
`float(np.max(np.abs(window)))` — a single realised log return of a
licensed-derived series, reproduced verbatim, and a sequence of them over
overlapping windows discloses actual return magnitudes and brackets their
dates. That reasoning was right and is unchanged.

**The same values were already committed in markdown, under a different
label** — and, it turned out, so were four raw index prices and three realised
returns. Refusing a parquet while shipping its contents in prose is incoherent,
and Stooq's terms carry no de minimis clause (docs/data_licenses.md).

## The rule, stated once

> An **aggregate** over many observations is publishable. A function that
> **returns one realised observation** is not.

Publishable: a mean, a variance, a kurtosis, a skew, a count, a fitted
parameter, a log-likelihood, a forecast, a loss, and any ratio of two of these.
Not publishable: a max, a min, "the largest day", a quoted price, return or
target value, or a one-bar estimator, which is a realised observation of a
derived series.

**Ratios and ranks, not coarsening.** Every argument these numbers were carrying
is comparative, so the comparison survives without the value. Rounding
`0.14183` to `0.142` would still hand over three digits of a real observation —
enough to match against the underlying series — whereas a ratio discloses
neither operand. Where a denominator is itself an order statistic (a cell's
median window maximum), it is never published either, so the ratio identifies
nothing on its own.

**Forecasts, losses and fitted parameters stay as levels.** They are the study's
own output rather than observations of the series, and no model in the grid
emits a raw observation — `naive`'s forecast is a trailing RMS, `ewma`'s an
exponentially weighted mean, the rest are fitted. That is why
`docs/P3_ANALYSIS_VALIDITY.md` still quotes `forecast min` while its
`proxy min` column became a ratio.

## 1. What was fixed, before and after

### docs/P3_CONVERGENCE_FORENSICS.md

| where | before | after |
|---|---|---|
| §4 prose | "Every fit from 4993 onward has the same window maximum, **\|r\| = 0.14183**" | "Every fit from 4993 onward sees the **same** window maximum — the ratio column of T5 does not move across the run" |
| T4 row | `window max \|r\|` — `+0.0294 … +0.1418` across eight quantiles of maxima, several of which *are* maxima | `window max \|r\| / clean median` — `+0.5039 … +2.4340`, the same eight quantiles as ratios to the cell's clean-fit median |
| T5 column | `window max \|r\|`, 23 rows, **beside a `date` column** — 23 (magnitude, dated) pairs, and `0.14183` recurring because it is one return seen in seventeen windows | `window max \|r\| / clean median`, same 23 rows; the constancy from origin 4993 is still visible as an unmoving `2.434` |
| §5 prose | "larger … by a factor between **1.106** (0.23980 vs 0.21688, origin 541) and **1.801** (0.17588 vs 0.09768, origin 2494)" | "larger … by a factor between **1.106×** (origin 541) and **1.801×** (origin 2494)" — the operands were four more realised returns |
| §5 prose | window standard deviation "(e.g. 0.0594 vs 0.0483 at origin 499)" | "between 1.101 (origin 2494) and 1.374 (origin 667)" — a std is an aggregate and would have been publishable as a level; given as a ratio to read alongside the row above |
| T6 columns | `BTC max \|r\|` and `ETH max \|r\|`, 15 rows each, beside a `date` column | one `ETH/BTC max \|r\|` column, the ratio the comparison was always about |

Lines 402 and 408's counted forms — `0 of 15` / `15 of 15`, and "larger at
every one of the fifteen" — were already correct and were the model for the
rest. T1, T2, T3, T7 and T8 were read and carry only fitted parameters,
optimiser state, dates, counts and rates; nothing changed in them. §8's "the
largest fitted ν in the grid is 50.000000" is a fitted parameter sitting on its
own box bound, not an observation.

### docs/P3_ANALYSIS_VALIDITY.md

| where | before | after |
|---|---|---|
| §1.1 table | `proxy min (>0)` — eleven smallest strictly positive realized targets, verbatim (`2.431e-10`, `6.716e-11`, `4.401e-11` …) | `proxy min / median` — each over that asset's median positive target, medians taken over 3,291 to 5,510 values |
| §1.3 table | **five raw index prices per row** (`prev C`, `O`, `H`, `L`, `C`) for three dated bars, plus `ln(C/C_-1)` as **+4.13% / -1.41% / +1.90%**, plus a one-bar Parkinson estimate | the bar's *shape* (`stale open`, `O = L, C = H`, direction) and two ratios: `\|ln(C/C_-1)\| / median \|r\|` (**5.6× / 1.9× / 2.6×**) and `Parkinson / median Parkinson` (**11.86× / 1.39× / 3.42×**) |
| §1.3 prose | "a target of exactly 0.0 on a day the index moved several percent" | "… several times its own median daily move — 5.6× on the first of the three" |
| §1.4 table | `target` — five realised target values beside their exact dates | `target / asset median target` |
| §1.4 prose | NKX 2020-10-01 "whose four prices span 19 index points" | "whose high-low range is **one thousandth** of that asset's median daily range" |
| §5 summary | "the largest carries a 4.1 % close-to-close return" | "the largest moved 5.6 times its asset's median daily move" |

The exact-zero targets stay as `0`: zero is the fact being reported and
discloses nothing. `forecast min`, `min/max forecast var`, `max proxy/forecast`,
the QLIKE ranges and the largest QLIKE terms all stay as they were, for the
reason given above.

### docs/PANEL_REPORT.md — found by the sweep, not named in the prompt

| where | before | after |
|---|---|---|
| §3, the zero-target example | "e.g. HSI 2024-04-12, `O=H=17095.03`, `L=C=16721.69`, previous close `17095.03`" | "e.g. HSI 2024-04-12, where `O = H`, `L = C` and the open printed at the previous close" |

Four raw index prices in one sentence. It is the same disclosure as
P3_ANALYSIS_VALIDITY §1.3 and was reached only by scanning every committed
document for price-shaped numbers rather than by following the two starting
points.

## 2. Every argument the documents made, they still make

| the claim | what carried it before | what carries it now |
|---|---|---|
| One unchanged extreme return dominates HSI's late `gjr` run | the same `0.14183` repeated down T5 | the same `2.434` repeated down T5, and T4's fallback min = median = max |
| That run's windows are far more extreme than the cell's typical window | fallback max `0.1418` vs clean median `0.0583` | fallback ratio `2.434` vs clean median ratio `1.000` |
| ETH's windows carry larger extremes than BTC's at every one of BTC's fallback origins | two columns of maxima to compare by eye | one ratio column, every entry > 1, range 1.106–1.801 |
| CAC, NKX and TWSE have targets that approach zero closely enough to threaten QLIKE | `2.431e-10`, `6.716e-11`, `4.401e-11` | `3.479e-06`, `3.491e-07`, `9.465e-07` of their own medians — six to seven orders of magnitude below a typical day |
| The zero-target days are large moves, not quiet days | a quoted `+4.13%` return and five prices | `5.6×` the asset's median daily move, on a monotone bar with a stale open |
| The near-zero scored days are extreme relative to the asset | five raw target values | five ratios spanning `9.5e-07` to `3.8e-05` of the asset's median |

Nothing was coarsened and no claim was weakened. Two claims are now *easier* to
read: T4's "the fallback windows are the extreme ones" is a ratio against 1.000
rather than two levels to divide mentally, and §1.1's three at-risk assets are
visible as six orders of magnitude rather than as raw exponents that depend on
each index's price scale.

## 3. The rule is now structural

It lived in a document, which is why it took a second prompt to notice it had
been broken in three tables. It now lives in
`volbench.benchmarks.convergence_forensics.COLUMN_POLICY`: every derived column
tagged `aggregate` or `order_statistic` at construction, and every emitting path
through one gate.

| piece | what it does |
|---|---|
| `COLUMN_POLICY` | 57 columns, each tagged. `max_abs_return` is the only `order_statistic` |
| `unpublishable_columns` | order statistics **and untagged columns** — fail closed, so a column added later cannot reach a report by nobody having thought about it |
| `refuse_unpublishable` | raises `OrderStatisticError` naming the columns |
| `publishable` | drops the order statistics, then still runs the check, so the convenience cannot become a bypass |
| `with_publishable_ratios` | adds `max_abs_over_clean_median`, the ratio that carries the argument without the value |
| `write_fits` | the artifact writer — refuses before it opens the file |

`gamma_persistence_table` (T4), the new `fit_run_table` (T5) and
`paired_windows` (T6) all build through `publishable`/`refuse_unpublishable`,
so the tables cannot regain the column by someone adding it back to a list. The
same declaration governs the parquet: `main()` now writes the **publishable**
frame to `--out` (default `docs/P3_CONVERGENCE_FITS.parquet`) through
`write_fits`, and the full frame — order statistics included — to `--full-out`,
default `data/grid_primary/convergence_fits_full.parquet`, where the licensing
guard already governs by location. One declaration, two destinations, no
per-file judgement.

**Inert-proof.** `tests/test_order_statistics.py::TestTheWriterRefuses` writes a
clean frame successfully, then **re-tags an ordinary column
(`kurtosis`) as an order statistic** and requires the same writer to refuse the
same frame — and to leave no file behind. It is the tag that is load-bearing,
not one hard-coded column name. Two further halves keep the gate honest: an
untagged column is refused like an order statistic, and a clean frame is
written rather than refused.

## 4. What the tables were regenerated from

The existing `docs/P3_CONVERGENCE_FITS.parquet`, on disk from the earlier run.
**Nothing was re-fitted and no store was touched** — all 374 files of
`data/grid_primary/store/` are identical in SHA-256, size and mtime to before
this branch.

Every column other than the redacted one reproduces the committed table
character for character, which is the check that the regeneration is faithful
rather than a fresh computation wearing the old numbers' clothes:

| table | checked | result |
|---|---|---|
| T4 | 8 rows | 7 byte-identical; the 8th is the row that was replaced |
| T5 | 23 rows × 10 remaining columns | **0 differences** |
| T6 | 15 rows × 8 remaining columns | **0 differences**, and every new ratio equals `ETH_max / BTC_max` of the two values it replaced to 4 dp |

The parquet itself was then projected in place — the working frame with
`max_abs_return` written to `data/grid_primary/convergence_fits_full.parquet`,
the publishable 49-column frame back to `docs/`. That resolves the first of the
two blockers docs/P3_REPO_HYGIENE.md §3 named. **The second still stands:**
`tests/test_licensing_guard.py::TestNoDataIsTracked::test_no_archive_or_parquet_data_files_anywhere`
refuses any tracked `.parquet` anywhere, so the file remains uncommitted and
`git add -A` still stages it into a failing guard. Whether that guard should
become location-shaped is the same decision N left open, and it is not this
branch's either.

## 5. Anything left carrying an order statistic

**Nothing, in any committed document.** Every tracked markdown file was scanned
for superlative language (`largest`, `smallest`, `maximum`, `minimum`,
`highest`, `lowest`, `extreme`), for price-shaped numbers, and for table rows
pairing a date with a value. What the scan surfaced and why each remaining hit
is publishable:

| document | hit | why it stays |
|---|---|---|
| `P3_LOSS_TABLES.md` | "maximum 29 %" | a max over *changes in standard errors* — an aggregate of aggregates |
| `P3_PAIRWISE_COMPLETE.md` | `largest_drop`, `largest_off_diagonal_drop` | counts of dropped origins |
| `P3_METRIC_TARGETS.md` | "a maximum of 11×" | a ratio of two model quantities |
| `P3_LGBM_SMEARING_AUDIT.md` | "the smallest gap (1.07)", "residual scale … smallest (0.34–0.39)" | smearing factors and residual scales — model quantities |
| `P3_TSFM_VARIANCE_AUDIT.md` | "median coefficient of variation is the lowest" | an aggregate |
| `P3_GRID.md` | "slowest single cell: CAC/patchtst, 3.5 min" | a timing |
| `PANEL_REPORT.md` | "TWSE's 13-day maximum" gap; "smallest: BTC-USD at 2792" | a gap length in days and a count of observations |
| `P3_ANALYSIS_VALIDITY.md` | `forecast min`, `max proxy/forecast`, largest QLIKE terms | forecasts, ratios and losses — the study's own output |
| `P3_CONVERGENCE_FORENSICS.md` | "the largest fitted ν in the grid is 50.000000" | a fitted parameter on its box bound |

Two things are recorded rather than fixed, because neither is a committed
document and neither is this branch's call:

- **`docs/P3_CONVERGENCE_FITS.parquet` is still uncommitted**, on the licensing
  guard's blanket `*.parquet` rule described in §4.
- **Git history** still contains the removed values, in the versions of these
  documents committed before this branch. Scrubbing history is a packaging
  decision about what the release repository is built from — the same class as
  the author identity docs/P3_REPO_HYGIENE.md §4 left open — and rewriting
  published history to fix it here would be worse than recording it.
