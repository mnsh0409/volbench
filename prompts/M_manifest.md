# Prompt M — Manifest provenance: one current manifest, committed, and a guard so it stays that way

**Terminal:** the main integration checkout, on a new branch off `feat/p3-analysis`:

```bash
git switch feat/p3-analysis && git pull
git switch -c fix/manifest-provenance
```

**Session:** fresh and separately named — `claude -n vb-m-manifest`. Run `/effort high`.

**Size:** small and mechanical. It changes no number and re-runs no cell. Run it **before J3**, so J3 reads the right file by default rather than being told where to look.

---

## The problem, and why it is not a one-off

After the L fix run, the authoritative grid manifest is **`data/grid_primary/manifest_fix.json`** — under `data/`, therefore gitignored, therefore **uncommitted**. The committed `docs/P3_GRID_manifest.json` is now **stale**: it names the pre-fix hashes for the 44 cells L replaced.

The store is append-only, so both fragment sets are on disk. The manifest is the only thing that says which is current. Right now a clean checkout cannot tell, and the file that could tell it is not in the repository.

This is the **third** instance of one pattern:

1. The study driver lived under `data/` and was uncommitted — closed by J1 §4.
2. The data digests existed only in gitignored sidecars — closed by L's fold-in.
3. The fix-run manifest lives under `data/` and is uncommitted — open, and reopened *by* the run that closed (2).

So the fix is not "commit this file." Run outputs default to `data/` and reach `docs/` only when somebody notices. **Make the default correct and make the gap detectable**, or a fourth instance arrives with the next run.

---

## 0. The picture is worse than I described — start by classifying what exists

I wrote §1 below believing there were two manifests. There are **at least seven**:

- `docs/P3_GRID_manifest.json` — committed, and stale
- `data/grid_primary/manifest_preflight.json`
- `data/grid_primary/manifest_primary.json`
- `data/grid_primary/manifest_fix.json`
- `data/grid_primary/manifest_resume_after_k.json`
- `data/grid_primary/manifest_resume_after_move.json`
- `data/grid_primary/manifest_resume_after_fix.json`

plus copies mirrored in `../volbench-exports/grid_primary_2026-08-27/`.

Nothing in those names says which one a reader should believe. `manifest_fix.json` and `manifest_resume_after_fix.json` are indistinguishable to anyone who was not present when they were written, and several are plainly verification artifacts from resume runs rather than grid manifests at all.

So **before promoting anything**, produce `docs/P3_MANIFEST_INVENTORY.md`: one row per manifest file, giving its path, its digest, when it was written, how many cells it names, the digest of the fragment set it points at, and — in one clause — **what it is** (grid manifest for a named run / resume-verification artifact / superseded copy / export mirror).

Then state which single file is the current grid manifest, and **how you established that** rather than which name sounds most recent. Two independent checks, at minimum: that its cell hashes resolve to fragments present in the store, and that those fragments are the ones the post-fix cells should have (44 changed by L, 99 unchanged).

If two candidates both satisfy that, say so and stop — do not pick one. Everything in §1 depends on this answer being right, and "the file with `fix` in the name" is not an answer.

The export-bundle copies stay where they are and are not promoted: that bundle is internal-transfer-only and carries values derived from Stooq and Binance data. Note their existence in the inventory and leave them alone.

## 1. One current manifest, committed

- **`docs/P3_GRID_manifest.json` is always the current grid manifest.** Promote the post-fix manifest into it.
- **Archive the superseded ones** rather than deleting them — plural, per §0 — J1's evidence documents are anchored to the pre-fix grid and must stay checkable. Put it somewhere unambiguous (`docs/archive/` or a dated filename), and make clear from the name alone that it is not current.
- Add three fields to the current manifest so it is self-describing:
  - `supersedes` — the digest (and archived path) of the manifest it replaces
  - `store_digest` — the digest of the fragment set it names
  - `manifest_digest` — its own digest, computed over the rest of the file

The last one matters more than it looks. Provenance in this project is being **re-anchored on content digests rather than commit SHAs**, because the public release repository will not carry the development history. A manifest that names its own digest is citable from a paper and verifiable by recomputation; a commit SHA in a repository nobody can browse is not.

Check whether the interpreter version is present (L folded that in). If it is not in the post-fix manifest, add it — J1 had to establish CPython 3.11.5 three indirect ways because the run recorded it nowhere.

## 2. Make the default correct

Change the grid driver so the **manifest** is written under `docs/` by default. Only the **store** — the parquet fragments and their sidecars — stays under `data/`.

The distinction is real and worth stating in the code: the store holds values derived from licensed market data and must never be tracked; the manifest holds hashes, versions, environment pins, timings and cell status, and holds **no series values**. That is the same argument that let `docs/P3_GRID_manifest.json` be committed in the first place. Verify it rather than assuming it — enumerate the manifest's contents and confirm no field carries a price, a return or a variance.

## 3. A guard, with an inert-proof

Add a test asserting that **no manifest exists under `data/` without a committed counterpart under `docs/`.**

Then prove the test is not vacuous. Once the driver writes to `docs/`, there will be no manifest under `data/` at all, and a check that passes because there is nothing to check is not a check — that is D-042's whole point. So the test needs a fixture that plants a manifest under `data/` and asserts the guard **fires**. Both halves, or it is decoration.

Keep it distinct from `tests/test_licensing_guard.py::TestNoDataIsTracked`, which answers a different question ("is anything under `data/` tracked?") and must stay exactly as it is.

## 4. Verify

Report all four:

- `git ls-files -- data/` still empty; the licensing guard green
- The resumability re-run still reports **143 cached, 0 computed, 0 failed**, fragments byte-identical and unrewritten
- The guard test passes **and** its inert-proof fires
- Full gate: `ruff`, `mypy --strict`, tests on 3.11/3.12/3.13; push the branch so CI sees it before any merge

## 5. Report

The inventory table from §0 and how you established which manifest is current; the path and digest of the current manifest; the archived paths and digests of the superseded ones; the driver default before and after; the guard's two results; the four verifications above. Note explicitly whether the manifest contained any series values (it should not) and how you established that.

**Change nothing else.** No re-runs, no numbers, no model code. If you find another output that defaults to `data/` and should not, write it down and leave it.
