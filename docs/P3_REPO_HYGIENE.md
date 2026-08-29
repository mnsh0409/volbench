# P3 — four repo-hygiene items, none of which moves a number

Nothing here changes a forecast, a score or a config hash. The one thing that
does change is a *content digest*, in §4, deliberately and once; it is reported
in full there. Task 3 stopped short of its commit, and §3 says why.

---

## 1. CI now runs on `fix/**`

`.github/workflows/ci.yml` triggered pushes on `main`, `feat/**` and `m2/**`.
Two branches had shipped without CI ever running — `fix/p3-model-defects` and
`fix/manifest-provenance` — and in both cases the absence was noticed only by
asking the Actions API for the branch's run count and getting `0`. The workflow
file's own comment already explains why the list exists: this project merges
branches without pull requests, so the **push trigger is the CI gate**. `fix/**`
was missing because no such branch existed when the list was written.

```diff
-    branches: [main, "feat/**", "m2/**"]
+    branches: [main, "feat/**", "fix/**", "m2/**"]
```

`m2/**` is dead now that Phase 2 is complete and was **left alone**: removing it
in this commit would muddy the record of what this commit is for.

GitHub evaluates the workflow file at the pushed commit, so this change
self-triggers. Confirmed rather than assumed — see §5.

## 2. Three analysis entry points named a gitignored manifest

`benchmarks.loss_tables`, `benchmarks.defect_tables` and
`benchmarks.convergence_forensics` each defaulted `--manifest` to
`data/grid_primary/manifest_fix.json`. That file is gitignored, so a clean
checkout could not run any of the three even though docs/P3_MANIFEST_INVENTORY.md
had just committed the same manifest one directory away. All three now default
to `docs/P3_GRID_manifest.json`.

**It changes no number, and here is the proof rather than the claim.**

*The two files are the same manifest.* Same `manifest_digest`
(`26842732cb6e98fc…`, as it stood before §4), the same 143 config hashes **in
the same order**, and every per-cell field identical except `wall_clock_s`:

```
committed  manifest_digest: 26842732cb6e98fc1ee756e5533d7e19bbe17be88859d0bea5a2fd656caaeb2f
data/ fix  manifest_digest: 26842732cb6e98fc1ee756e5533d7e19bbe17be88859d0bea5a2fd656caaeb2f
cell hash lists identical, in order: True | n = 143
cell fields that differ (excluding wall_clock_s): none
```

*And the outputs are the same bytes.* `loss_tables` was re-run against
`docs/P3_GRID_manifest.json` **before** any prose was touched, and every one of
its four committed outputs came back byte-for-byte identical — `git status
docs/` reported nothing at all:

| file | SHA-256 before | after re-run |
|---|---|---|
| `docs/P3_LOSS_TABLES.md` | `dae0355a06714ac7…` | identical |
| `docs/P3_LOSS_TABLES.csv` | `de95d34c79e7009a…` | identical |
| `docs/P3_PAIRWISE_COMPLETE.md` | `09365179 8ff85d16…` | identical |
| `docs/P3_PAIRWISE_COMPLETE.csv` | `3d5bdc7f1bfbf8fc…` | identical |

`benchmarks.defect_tables` writes no file — it prints and returns L's
acceptance status — so it was run as a second check rather than diffed: against
the new default it still reports `acceptance (11 assets): PASS`, with panel
medians `smear_shipped 1.703` against `smear_realized 1.678`, the same numbers
L certified. `benchmarks.convergence_forensics` was not re-run: it re-fits all
7,101 GARCH-family fits, and §3 leaves its output uncommitted in any case.

Only then were the defaults and the documents' provenance prose changed, and
the grid regenerated. **Both CSVs — every number — stayed byte-identical**; the
only diffs in the regenerated markdown are the `| Grid |` provenance rows and
the "Which grid this is" paragraph, which had claimed that
`docs/P3_GRID_manifest.json` "describes the **pre-fix** grid". Since the
promotion it does not, and that sentence was the exact hazard this task exists
to remove.

### The audit that changed meaning without changing text

`benchmarks.tsfm_distribution_probe` already defaulted to
`docs/P3_GRID_manifest.json` and still does — so when that file was promoted
from the pre-fix to the post-fix grid, **the probe's default silently changed
which experiment it measures**, and nothing said so. `docs/P3_TSFM_VARIANCE_AUDIT.md`
now records, at the top, that its numbers are the pre-fix grid (run digest
`cb28a214`, store digest `8f1f83db`), that the closure it argued for has since
landed, and that reproducing it needs
`--manifest docs/archive/P3_GRID_manifest.91ba622a8e50.json`.

## 3. `.gitignore` ignores by location now — but the parquet was **not** committed

The blanket `*.parquet` rule is gone. `tests/test_licensing_guard.py` is built
on **location, not content**, precisely so that no per-file judgement is ever
needed, and a type-based ignore reintroduces exactly that judgement while
silently withholding evidence. Everything a study writes goes under `/data/` or
`results/`, and both rules remain, so every parquet that must never be
committed is still ignored — verified against `data/cache/…`,
`data/grid_primary/store/…`, `data/toy_benchmark/…` and `results/…`. A parquet
anywhere else is now *visible* rather than hidden, and whether it may be
committed is decided by reading its columns.

**`docs/P3_CONVERGENCE_FITS.parquet` was read, and it is not committable as it
stands.** 7,101 rows × 49 columns, enumerated in full:

| kind | columns | verdict |
|---|---|---|
| identity | `asset`, `config`, `fit_origin`, `date`, `window_start_date`, `crisis_tag` | fine — labels, an origin index and calendar dates |
| optimiser | `loglikelihood`, `convergence_flag`, `message`, `iterations`, `function_evals`, `scale`, `n`, `fallback`, `stored_status`, `refit_status` | fine — exit state and fit diagnostics |
| fitted parameters | `omega`, `alpha[1]`, `beta[1]`, `gamma[1]`, `nu`, each `_low`/`_high` bound, each `slack_*`, `alpha_plus_beta`, `persistence`, `omega_return_scale` | fine — estimates and their bounds |
| window moments | `std`, `kurtosis`, `skew`, `n_zero_returns` | fine — aggregates over 500 observations, not invertible to any of them |
| **window extremum** | **`max_abs_return`** | **series value — stop** |

`max_abs_return` is `float(np.max(np.abs(window)))` — literally the largest
absolute log return in each 500-day fit window, verbatim, to full float64
precision. It is an order statistic, not an aggregate: across 240 overlapping
origins per asset, every time it moves it discloses one more actual return
magnitude, and the ones it discloses are the extremes — the most identifiable
observations in a series whose redistribution Stooq's terms forbid
(docs/data_licenses.md). The other four moment columns cannot be inverted to an
observation; this one *is* one.

So the file stays out, per the standing rule, and this is reported rather than
worked around. Two things would have to happen before it can be committed:

1. **Drop or coarsen `max_abs_return`** in `benchmarks.convergence_forensics`
   — bucketing it, or replacing it with its ratio to the window `std`, would
   keep the diagnostic and disclose no observation. That regenerates the file
   and is a numbers-adjacent change this branch may not make.
2. **`tests/test_licensing_guard.py::TestNoDataIsTracked::test_no_archive_or_parquet_data_files_anywhere`
   would also refuse it.** That guard rejects *any* tracked `.parquet`
   anywhere, so it is the second, independent blocker — and it is content-shaped
   in exactly the way this task narrowed `.gitignore` away from. Whether it
   should become location-shaped too is the same decision, and it is not this
   branch's.

The file is now untracked-and-visible rather than untracked-and-hidden, which is
the part of the task that was safe to finish.

**One live consequence, stated so it is not a surprise:** until `max_abs_return`
is dealt with, a `git add -A` in this repository *will* stage
`docs/P3_CONVERGENCE_FITS.parquet`, and the licensing guard will fail with
`data-shaped files are tracked: ['docs/P3_CONVERGENCE_FITS.parquet']`. That is
the guard working — it is the only thing standing between a convenience command
and a licensing breach — but it is friction that lasts exactly as long as the
column does. Re-adding a targeted ignore would remove the friction by restoring
the hiding this task removed, so it was not done.

## 4. Identity leakage: a committed, shape-based test — and what it found

IJF review is double-blind and the reproducibility package is part of what a
reviewer sees. `tests/test_identity_leakage.py` sweeps every **tracked** file
for three shapes:

| shape | pattern |
|---|---|
| POSIX home directory | `/(?:home\|Users)/[A-Za-z0-9._-]+/` |
| Windows user directory | `[A-Za-z]:\Users\[A-Za-z0-9._-]+` |
| email address | `[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2}` |

**No name and no address appears in the test.** A test that greps for a
particular person must contain that person's name, ships publicly, and leaks
exactly what it was written to prevent. Shapes catch the real problem and reveal
nothing. For the same reason the fixtures are assembled from fragments at run
time rather than written out, and a test asserts that **the file does not match
its own patterns** — because excluding the sweep's own source from the sweep
would put a blind spot precisely where someone hiding something would put it.

**Inert-proof:** four planted identities, one per shape (POSIX `/home/…` and
`/Users/…`, Windows, email), each asserted to be found and correctly named; plus
a test that the report carries the file and line, a test that ordinary absolute
paths and version strings are *not* flagged, and a test that a binary file does
not break the sweep. All pass.

### What it found — seven leaks in seven files, all fixed

| file | shape | was | now |
|---|---|---|---|
| `docs/P3_GRID_manifest.json` | POSIX home | `environment.interpreter.executable` | field removed |
| `docs/archive/manifest_resume_after_fix.*.json` | POSIX home | same field | field removed, file renamed |
| `docs/archive/P3_GRID_manifest_resume_after_m.*.json` | POSIX home | same field | field removed, file renamed |
| `docs/P3_MANIFEST_INVENTORY.md` | POSIX home | the path quoted in prose | paragraph rewritten |
| `src/volbench/data/crypto.py` | email | contact address in the `User-Agent` | `$VOLBENCH_CONTACT`, unset by default |
| `src/volbench/data/stooq.py` | email | same | same |
| `tests/test_results.py` | POSIX home | `Path("/home/…/data")`, used only as *an absolute path* | `Path("/srv/volbench/cache")` |

The two adapters kept the courtesy contact available to whoever runs them —
setting `VOLBENCH_CONTACT` to an address puts `(+mailto:…)` back in the header —
and ship with it empty, so the header still names the tool and its version, which is
the part the endpoint needs. No test pinned the old string.

**Fixed at the source, not only in the artefacts.** `determinism.interpreter_info`
no longer emits `executable` at all, so no future run records it. It never
reached a config hash — `environment_spec` carries only `blas_threads` (D-032),
and every store sidecar's `environment` block is `{"blas_threads": 1}` — so
removing it cannot move a cell. Verified before the edit, not after.

### The digest that moved

Removing a field changes `manifest_digest`, which docs/P3_MANIFEST_INVENTORY.md
made a citable anchor. Done once, deliberately, with every citation updated in
the same commit:

| | before | after |
|---|---|---|
| `docs/P3_GRID_manifest.json` `manifest_digest` | `26842732cb6e98fc1ee756e5533d7e19bbe17be88859d0bea5a2fd656caaeb2f` | **`4559a7033bc52c741a56c58fbdad7d584db75e1f7018ee7dcdd6379b18d534e8`** |
| `store_digest` | `05efdb459c9e95ce…` | **unchanged** — no fragment moved |

The redaction was applied to **every copy of every affected manifest**, the
gitignored `data/` originals included, so `orphaned_manifests`' counterpart
relation still holds and no manifest under `data/` became stranded:

| manifest | file SHA | run digest | store digest |
|---|---|---|---|
| `docs/P3_GRID_manifest.json` | `4e217db2` | `4559a703` | `05efdb45` |
| `data/grid_primary/manifest_fix.json` | `6ab20fc5` | `4559a703` | `05efdb45` |
| `docs/archive/manifest_resume_after_fix.4590cc1e836b.json` | `4590cc1e` | `751411d1` | `05efdb45` |
| `data/grid_primary/manifest_resume_after_fix.json` | `4590cc1e` | `751411d1` | `05efdb45` |
| `docs/archive/P3_GRID_manifest_resume_after_m.d4f14a90f33d.json` | `d4f14a90` | `6a4fa62b` | `05efdb45` |

The two archived files were renamed because their content-addressed names
asserted a file digest that had stopped describing them. The four pre-L
manifests carry no `interpreter` block and were not touched.

### Identity the shape patterns cannot see — reported, not changed

These are real double-blind exposures and none is matchable by shape, only by
naming a person, which is the thing the test must not do. They are a decision,
not a defect:

- **`pyproject.toml`** — `authors = [{ name = "..." }]` and
  `Repository = "https://github.com/<handle>/volbench"`. The released package
  needs *some* author and *some* URL, so what these should say during review is
  a submission decision.
- **`SETUP.sh`** — an example data root and an example GitHub login, both built
  from the author's name.
- **`docs/decisions.md`** (D-008) and **`docs/research_design.md`** — name the
  author in prose. Both are read-only mirrors here (CLAUDE.md); flagged for the
  planning machine, not edited.
- **Git history itself** — author name, email and commit trailers. Out of reach
  of any file-level check, and a packaging question rather than a repository one.

## 5. The gate

Placed here because §1's confirmation and the resumability re-run are both
statements about the branch as a whole rather than about any one task.

| leg | result |
|---|---|
| resumability re-run (`--tag resume_after_n`) | **143 cached, 0 computed, 0 failed** |
| store SHA-256, all 374 files, before vs after | identical |
| store size + mtime, all 374 files | identical — **not rewritten at all** |
| that run's `store_digest` | `05efdb45…`, unchanged by anything on this branch |
| that run's `environment.interpreter` | `{"python": "3.11.5", "implementation": "CPython"}` — **no `executable`**, so §4's fix holds at the source and not only in the artefacts |

The run's manifest is archived at
`docs/archive/P3_GRID_manifest_resume_after_n.e217cb473269.json` (row 8 of
docs/P3_MANIFEST_INVENTORY.md) and the working copy removed, as
docs/P3_MANIFEST_INVENTORY.md §4 prescribes for a verification run.

| leg | result |
|---|---|
| `pytest`, Python **3.11.5**, `--extra classical` | **1,371 passed**, 0 failed, 29 skipped |
| `pytest`, Python **3.12**, `--extra classical` | **1,400 collected, 1,365 passed**, 0 failed, 0 errors, 35 skipped |
| `pytest`, Python **3.13**, `--extra classical` | **1,400 collected, 1,365 passed**, 0 failed, 0 errors, 35 skipped |
| `ruff check .` on all three | clean |
| `mypy` (strict, `src`) on all three | no issues in 55 source files |
| `git ls-files -- data/` | empty; the licensing guard green (13 passed) |

The 3.11 leg carries the `tsfm` extra and 29 rather than 35 skips because the
foundation-model adapter tests can import their backends there; the extra six
skips on 3.12/3.13 are those tests, which have no torch in those venvs.

**CI ran, and this is the whole point of §1.** The push of `5e3641e` to this
`fix/**` branch triggered the workflow at that commit, where before the same
query returned `total_count: 0`:

```
$ gh api repos/<repo>/commits/5e3641ea.../check-runs
  test (3.11)    completed  success   08:24:41Z -> 08:42:28Z
  test (3.12)    completed  success   08:24:41Z -> 08:43:17Z
  test (3.13)    completed  success   08:24:41Z -> 08:36:44Z
```

Three legs, `--extra classical --extra torch-cpu`, all green. Confirmed against
the Actions API rather than assumed — which is the lesson the two branches that
shipped without it paid for.
