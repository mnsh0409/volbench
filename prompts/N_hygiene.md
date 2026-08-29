# Prompt N — Four repo-hygiene items, none of which moves a number

**Terminal:** the main integration checkout, on a new branch off `feat/p3-analysis`:

```bash
git switch feat/p3-analysis && git pull
git switch -c fix/repo-hygiene
```

**Session:** fresh and separately named — `claude -n vb-n-hygiene`. Run `/effort high`.

**Scope rule:** nothing here changes a forecast, a score, or a config hash. If a change would, stop and report instead of making it. Task 4 is the exception in one narrow respect, and it says so.

---

## 1. CI does not run on `fix/**`

`.github/workflows/ci.yml` triggers pushes on `main`, `feat/**`, `m2/**`. Two branches have now shipped without CI — `fix/p3-model-defects` and `fix/manifest-provenance` — and in both cases the absence was discovered only by querying the Actions API for `total_count: 0`.

Add `"fix/**"` to the push trigger. The workflow file's own comment already explains why the list exists: this project merges feature branches without pull requests, so the push trigger *is* the CI gate. `fix/**` was omitted because no such branch existed when the list was written, not by intent.

Pushing this change should self-trigger, since GitHub evaluates the workflow file at the pushed commit. Confirm it actually ran rather than assuming — that is the whole lesson here.

`m2/**` is dead now that Phase 2 is complete. Leave it; a separate concern, and removing it in this commit muddies the record.

## 2. Three analysis entry points still name a gitignored manifest

`loss_tables`, `defect_tables` and `convergence_forensics` each default `--manifest` to `data/grid_primary/manifest_fix.json`, which is gitignored — so a clean checkout still cannot run them. Prompt M closed exactly this gap one layer down; these are the same gap one layer up.

Repoint all three at `docs/P3_GRID_manifest.json`.

**This must change no number, and you should prove it rather than assert it:** the two files carry the same `manifest_digest` and name the same 143 cells. Show that, then show that re-running one of the three against the new default reproduces its committed output byte-for-byte.

While you are there, `docs/P3_TSFM_VARIANCE_AUDIT.md` needs one line recording **which fragment set it was computed against**. `tsfm_distribution_probe` already defaulted to `docs/P3_GRID_manifest.json`, so its default silently changed meaning when that file was promoted — pre-fix to post-fix — and anyone re-running it now would compare post-fix numbers against a pre-fix document with nothing announcing the difference.

## 3. `.gitignore` ignores by file type where it should ignore by location

`docs/P3_CONVERGENCE_FITS.parquet` — 7,101 rows, the evidence behind the convergence forensics — is untracked, caught by a blanket `*.parquet` rule rather than by `/data/`.

That rule has the wrong shape. `tests/test_licensing_guard.py` is deliberately built on **location, not content**, precisely so no per-file judgement is ever needed. A type-based ignore reintroduces exactly the judgement call the location rule exists to remove, and it silently withholds committed-adjacent evidence.

- Narrow the rule so parquet under `data/` stays ignored and parquet elsewhere does not.
- Before committing the forensics parquet, **enumerate its columns and confirm it holds no series values** — fitted parameters, log-likelihoods, exit flags and origin indices are fine; a price, return or variance is not. Same standard M applied to the manifest, same method: enumerate, do not assume.
- If any column does carry series values, do not commit it. Report which, and stop.

## 4. A committed test for identity leakage — and a trap in writing it

IJF reviewing is **double-blind**: the manuscript and everything a reviewer sees must not identify the author. The reproducibility package is part of what a reviewer sees.

M correctly recorded, but did not treat as a problem, that `environment.interpreter.executable` in the committed manifest is an **absolute path under a named home directory**. That identifies the author. It is very unlikely to be the only instance — check `docs/P3_GRID.md`, `P3_DRIVER_PROVENANCE.md`, the evidence documents, any captured console output, and test fixtures.

**Add a test that fails when a tracked file contains an identifying path or address.** Match on *shape*, not on any particular person:

- POSIX home paths — `/home/<name>/`, `/Users/<name>/`
- Windows user paths — `C:\Users\<name>\`
- Anything matching an email-address shape

**The trap, and it is a real one: do not hardcode the author's name or email in the test.** A test that greps for a name contains that name, ships publicly, and leaks precisely what it was written to prevent. Shape-based patterns catch the actual problem and reveal nothing. If a name-specific sweep is ever wanted, it belongs in an uncommitted local script, not in the repository.

Give it an inert-proof, as M's guard has: a fixture containing a home path, asserting the check fires.

**Then fix what it finds.** For the manifest field specifically: the *version* is what a reader needs in order to reproduce; the path is not. Keep `version` and `implementation`, and either drop `executable` or reduce it to its basename.

**This is the one place in this prompt where something changes:** editing the manifest changes its `manifest_digest`, which M made a citable anchor and which M's own outputs reference. So do it **once, deliberately**, recompute the digest, and update every document that cites the old one in the same commit. Report the old and new digests explicitly.

---

## Report

Per task: what changed, and the evidence it changed nothing it should not have. Specifically — CI confirmed to have actually run on this branch; the byte-for-byte reproduction from task 2; the column enumeration from task 3; the shape-based patterns, the inert-proof result, the full list of identity leaks found, and the old/new `manifest_digest` pair from task 4.

Full gate on all three interpreters, `ruff`, `mypy --strict`, and the resumability re-run showing 143 cached with nothing rewritten. Push the branch; do not merge.
