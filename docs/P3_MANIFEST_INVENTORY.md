# P3 — one current grid manifest, committed, and a guard so it stays that way

**The problem.** The results store is append-only. After the L fix run it holds
**187 fragments**: the 143 the study scores and the 44 those replaced. Both sets
are on disk and neither is marked. The manifest is the only thing that says
which is which — and after L the authoritative manifest was
`data/grid_primary/manifest_fix.json`, under `data/`, therefore gitignored,
therefore uncommitted. The committed `docs/P3_GRID_manifest.json` still named
the pre-fix hashes for those 44 cells. A clean checkout could not tell the two
apart, and the file that could tell it was not in the repository.

That is the third instance of one pattern, not a one-off:

1. the study driver lived under `data/` and was uncommitted — closed by
   docs/P3_DRIVER_PROVENANCE.md;
2. the input digests existed only in gitignored sidecars — closed by L's fold-in
   into `docs/P3_DATA_DIGESTS.json`;
3. the fix-run manifest lived under `data/` and was uncommitted — closed here,
   and reopened *by* the run that closed (2).

Run outputs default to `data/` and reach `docs/` when somebody notices. So the
fix is not "commit this file": it is to make the default correct
(§3) and the gap detectable (§4).

---

## 1. What existed — every manifest, classified

Seven manifests inside the repository and three more in the export bundle, none
of whose names says which one a reader should believe. `manifest_fix.json` and
`manifest_resume_after_fix.json` are indistinguishable to anyone who was not
present when they were written.

`run digest` is `manifest_digest` and `store digest` is `store_digest`, both
defined in §2 and both recomputable from the committed files. `written` is when
the run wrote the file, not when a copy of it was made.

| # | path | file SHA-256 | run digest | store digest | cells | comp/cach/fail | written | what it is |
|---|---|---|---|---|---|---|---|---|
| 1 | `docs/P3_GRID_manifest.json` **(current)** | `4e217db2` | `4559a703` | `05efdb45` | 143 | 44/99/0 | 2026-08-28 23:26 | **the current grid manifest**: the L fix run, promoted and annotated |
| 2 | `docs/archive/P3_GRID_manifest.91ba622a8e50.json` | `91ba622a` | `cb28a214` | `8f1f83db` | 143 | 130/13/0 | 2026-08-26 16:00 | superseded: the pre-fix committed manifest, the primary run |
| 3 | `docs/archive/manifest_preflight.03e0ecf845b9.json` | `03e0ecf8` | `a669e766` | `45ffdfa1` | 13 | 13/0/0 | 2026-08-26 14:21 | superseded: the 13-cell preflight run |
| 4 | `docs/archive/manifest_resume_after_k.80fbdde748d6.json` | `80fbdde7` | `91925032` | `8f1f83db` | 143 | 0/143/0 | 2026-08-28 20:45 | resume-verification artifact (pre-fix grid) |
| 5 | `docs/archive/manifest_resume_after_move.cf25799a885a.json` | `cf25799a` | `4ae045fb` | `8f1f83db` | 143 | 0/143/0 | 2026-08-28 12:57 | resume-verification artifact (pre-fix grid), the driver move |
| 6 | `docs/archive/manifest_resume_after_fix.4590cc1e836b.json` | `4590cc1e` | `751411d1` | `05efdb45` | 143 | 0/143/0 | 2026-08-29 00:26 | resume-verification artifact (post-fix grid), L's own re-run |
| 7 | `docs/archive/P3_GRID_manifest_resume_after_m.d4f14a90f33d.json` | `d4f14a90` | `6a4fa62b` | `05efdb45` | 143 | 0/143/0 | 2026-08-29 13:15 | resume-verification artifact (post-fix grid), §5 below |
| 8 | `docs/archive/P3_GRID_manifest_resume_after_n.e217cb473269.json` | `e217cb47` | `f5c1558d` | `05efdb45` | 143 | 0/143/0 | 2026-08-29 16:01 | resume-verification artifact, docs/P3_REPO_HYGIENE.md §5; the first written without the identifying field |
| — | `data/grid_primary/manifest_fix.json` | `6ab20fc5` | `4559a703` | `05efdb45` | 143 | 44/99/0 | 2026-08-28 23:26 | the file row 1 was promoted from; same run digest, so row 1 accounts for it |
| — | `data/grid_primary/manifest_primary.json` | `91ba622a` | `cb28a214` | `8f1f83db` | 143 | 130/13/0 | 2026-08-26 15:37 | byte-identical to row 2 |
| — | `data/grid_primary/manifest_preflight.json` | `03e0ecf8` | `a669e766` | `45ffdfa1` | 13 | 13/0/0 | 2026-08-26 14:21 | byte-identical to row 3 |
| — | `data/grid_primary/manifest_resume_after_{k,move,fix}.json` | — | — | — | 143 | 0/143/0 | — | byte-identical to rows 4–6 |
| — | `../volbench-exports/grid_primary_2026-08-27/manifest_primary.json` | `91ba622a` | `cb28a214` | `8f1f83db` | 143 | 130/13/0 | 2026-08-27 21:22 | export mirror of row 2 |
| — | `../volbench-exports/grid_primary_2026-08-27/run/manifest_primary.json` | `91ba622a` | `cb28a214` | `8f1f83db` | 143 | 130/13/0 | 2026-08-27 21:21 | export mirror of row 2 |
| — | `../volbench-exports/grid_primary_2026-08-27/run/manifest_preflight.json` | `03e0ecf8` | `a669e766` | `45ffdfa1` | 13 | 13/0/0 | 2026-08-27 21:21 | export mirror of row 3 |

The `store digest` column is the useful one: **ten of the sixteen files reduce to
three fragment sets.** `8f1f83db` is the pre-fix 143, `05efdb45` the post-fix
143, `45ffdfa1` the preflight 13. Manifests that differ only in run bookkeeping
— which cells were recomputed and how long each took — agree exactly about
which fragments the study is made of, and the digest says so without anyone
having to read 143 rows.

The three export-bundle copies **stay where they are and are not promoted**:
that bundle is internal-transfer-only and carries values derived from Stooq and
Binance data. They are listed because they exist, and left alone.

## 2. Which one is current, and how that was established

**`data/grid_primary/manifest_fix.json` — promoted to
`docs/P3_GRID_manifest.json`, run digest `4559a703`, naming fragment set
`05efdb45`.** Established by the following, not by which name sounds most
recent.

**a. Every cell resolves.** All 143 config hashes have both a `.parquet`
fragment and a `.json` sidecar in `data/grid_primary/store/`. So does the
pre-fix manifest's 143 — the store is append-only and both sets survive, which
is the whole reason the manifest has to be the one that decides.

**b. The store closes exactly.** The store holds 187 fragment pairs (374
files). The post-fix and pre-fix manifests name 143 each and share 99, so their
union is 143 + 44 = **187, with no fragment named by neither.** There is no
third set, no partial run, and nothing orphaned.

**c. The 44 that moved are the 44 L changed.** Against the pre-fix manifest,
cell for cell, exactly 44 config hashes differ, and they are precisely
`lgbm` (11 assets, the out-of-fold smearing fix) and `chronos`, `timesfm`,
`moirai` (11 assets each, the lognormal tail closure) — 11 × 4, every asset,
no other model touched. That is L's arithmetic exactly: 44 recomputed, 99
cached, 0 failed.

**d. A re-run agrees.** §5's resumability re-run computed a `store_digest` of
`05efdb45…` independently, from the store as it stands, and it matches the
promoted manifest's recorded value.

**The two candidates question.** `manifest_resume_after_fix.json` also names
fragment set `05efdb45` and passes (a)–(c) — it names *the same 143 config
hashes in the same order*, with the same `environment` block. They are not two
answers to "which fragment set is current": they agree, and there is nothing to
choose between. They differ in exactly two per-cell fields, `status` (44
`computed` vs 143 `cached`) and `wall_clock_s`, which is precisely the
difference between the run that *produced* those fragments and the run that
*checked* they were already there. The grid manifest is the production run's:
its statuses and timings are true statements about how each fragment came to
exist, where the resume run's would record the study as having computed
nothing. The same reasoning classifies rows 4–7 as verification artifacts
rather than grid manifests.

### The two digests

Both are implemented in `volbench.benchmarks.manifest_provenance` and are
reproducible from the committed files, deliberately: **provenance in this
project is anchored on content digests rather than commit SHAs**, because the
public release repository will not carry this development history. A manifest
that names its own digest is citable from the paper and verifiable by
recomputation; a commit SHA in a repository nobody can browse is not.

`manifest_digest`
: SHA-256 of the manifest with the three provenance fields removed, serialised
  as UTF-8 JSON with sorted keys and no whitespace (`separators=(",", ":")`).
  Excluding the envelope is what makes it a statement about *the run*, so the
  annotated file and the bare one the driver wrote share it — which is the
  counterpart relation §4's guard needs.

`store_digest`
: SHA-256 of one line per cell, `"{config_hash}  {parquet SHA-256}  {sidecar
  SHA-256}"`, sorted by config hash, joined by `\n` with a trailing newline. It
  binds the manifest both to *which* fragments it names and to their bytes.

Check a tree against its own manifest:

    uv run python -m volbench.benchmarks.manifest_provenance --check

## 3. The three fields, and what the manifest does not contain

`docs/P3_GRID_manifest.json` now opens with its own provenance:

```json
{
  "manifest_digest": "4559a7033bc52c741a56c58fbdad7d584db75e1f7018ee7dcdd6379b18d534e8",
  "store_digest": "05efdb459c9e95ce4adbd77c38446de58bab891072d6a2012e9cd96b8ada2a98",
  "supersedes": {
    "manifest_digest": "cb28a214676edbefacd461ff4086cd2ac69424d94399f93e57d3a16974026e5f",
    "file_sha256": "91ba622a8e501cb2b99ccab50d3da809985866f3bbfb79ef31a63571f9156ac5",
    "archived_path": "docs/archive/P3_GRID_manifest.91ba622a8e50.json"
  },
  "n_cells": 143,
  ...
```

**The interpreter is already recorded** — `CPython 3.11.5`, folded in by L, in
`environment.interpreter`. J1 had to establish that three indirect ways because
the run recorded it nowhere; nothing needed adding here.

**It contains no series values.** That is the argument that let a manifest be
committed in the first place, so it is checked mechanically rather than
asserted. Every leaf of the JSON was enumerated by path and type. The result is
41 distinct paths:

| where | keys | what they are |
|---|---|---|
| top level | `n_cells`, `n_computed`, `n_cached`, `n_failed`, `n_fits`, `n_fits_fallback` | counts |
| `environment` | `blas_threads`, `cpu_count`, `thread_pin_explicit`, `kernel_signature`, `env.*`, `blas.*`, `interpreter.*`, `observed_thread_pools[].*` | pins, versions, thread counts, one interpreter path |
| `cells[]` | `index`, `horizon`, `n_rows`, `n_missing`, `n_fits`, `n_fits_fallback`, `n_fits_nonconverged` | integer counts and indices |
| `cells[]` | `asset`, `model`, `arm`, `lane`, `status`, `config_hash`, `error` | labels, a SHA-256, and `null` |
| `cells[]` | `wall_clock_s` | **the only float in the file** — a wall clock, 143 of them |

No price, no return, no variance, no series of any kind: the only floating-point
number anywhere is a timing, and the only data-derived values are SHA-256
config hashes, which are not the data. `tests/test_manifest_provenance.py::TestTheCommittedManifest::test_it_contains_no_series_values`
enumerates the cell keys and asserts the float is `wall_clock_s`, so this stays
true rather than having been true once.

**One field was removed after the fact, and the digest moved with it.** The
manifest as promoted also carried `environment.interpreter.executable`, an
absolute path under a home directory — so it named the person who ran the
study, and IJF review is double-blind. That was recorded here as a wart and
left; `docs/P3_REPO_HYGIENE.md` §4 removed it instead, from every copy at once,
and `determinism.interpreter_info` no longer produces it. The interpreter
*version* stayed: the version is what reproduces a run, the venv path is
meaningful only on the machine that already has it.

The redaction changed this manifest's citable anchor, once and deliberately:

| | before | after |
|---|---|---|
| `manifest_digest` | `26842732cb6e98fc…` | **`4559a7033bc52c74…`** |
| `store_digest` | `05efdb459c9e95ce…` | unchanged — no fragment moved |

Every citation of the old digest was updated in the same commit. The two
archived manifests that carried the same field were redacted with it, and
renamed, because their content-addressed filenames asserted a file digest that
had stopped describing them: `manifest_resume_after_fix.b1605a43ec03.json` →
`…4590cc1e836b.json`, and `P3_GRID_manifest_resume_after_m.367b9dbd3ba4.json` →
`…d4f14a90f33d.json`.

## 4. The default is now correct

`volbench.benchmarks.grid_primary` writes the manifest under `docs/` by
default — `--manifest-dir`, default `docs/` — and only the **store** stays
under `--out-dir` (`data/grid_primary`). The distinction is stated in the
driver's docstring because it is the whole rule: the store holds values derived
from licensed market data and can never be tracked (docs/data_licenses.md); the
manifest holds hashes, versions, environment pins, timings and per-cell status
and no series values, which is why it can be.

Three consequences, all in `main()`:

- **The default tag owns the canonical name.** `--tag primary` (the default)
  writes `docs/P3_GRID_manifest.json`; any other tag writes
  `docs/P3_GRID_manifest_<tag>.json`, which reads as the verification artifact
  it is. So the current grid manifest is what a default run *produces*, rather
  than something a human remembers to copy afterwards.
- **What it replaces is archived first**, before `run_grid` writes, and recorded
  in `supersedes`. Superseded manifests are never overwritten.
- **A restricted run is refused under the default tag.** `--assets`/`--models`
  make a run something other than the study, and its manifest must not end up
  wearing the whole grid's name. This hazard is new — it arrives with the
  canonical default — so it is guarded rather than left to convention.

The driver then annotates the manifest `run_grid` wrote with the three fields,
so a manifest is self-describing from the moment it exists.

| | before | after |
|---|---|---|
| manifest | `data/grid_primary/manifest_primary.json` (gitignored) | `docs/P3_GRID_manifest.json` (committed, annotated) |
| store | `data/grid_primary/store/` | unchanged |
| reports | `data/grid_primary/{summary,report}_primary.*` | unchanged — see §7 |

## 5. The guard, and the proof that it is not inert

`tests/test_manifest_provenance.py::TestNoManifestIsStrandedUnderData` asserts
that **no manifest exists under `data/` without a committed counterpart under
`docs/`**. Counterpart is by `manifest_digest`, not by file identity: the
committed manifest carries three fields the driver's file does not, and a
byte-identity relation would strand the very file a promotion came from.

It is kept distinct from `tests/test_licensing_guard.py::TestNoDataIsTracked`,
which asks a different question — is anything under `data/` *tracked* — and is
unchanged.

Once the driver writes to `docs/`, a clean tree has no manifest under `data/`
at all, and a check that passes because there is nothing to check is not a
check. So the guard is proved against a planted manifest rather than trusted:

| test | asserts |
|---|---|
| `test_the_repository_has_none` | the real tree is clean (skipped without a local store) |
| `test_the_guard_fires_on_a_planted_manifest` | **plants a manifest under a `data/` root and asserts the guard reports it** |
| `test_a_committed_counterpart_clears_it` | it does *not* fire on an accounted-for manifest — a guard that fires on everything gets switched off |
| `test_annotation_does_not_break_the_counterpart_relation` | an annotated committed manifest still accounts for the bare file |
| `test_it_looks_at_manifests_and_not_at_the_store` | fragment sidecars and `summary_*.json` are not manifests |

## 6. Verification

| check | result |
|---|---|
| `git ls-files -- data/` | empty |
| `tests/test_licensing_guard.py` | 13 passed |
| resumability re-run (`--tag resume_after_m`) | **143 cached, 0 computed, 0 failed** |
| store SHA-256, all 374 files, before vs after | identical |
| store size + mtime, all 374 files, before vs after | identical — not rewritten at all |
| re-run's independently computed `store_digest` | `05efdb45…`, matches the promoted manifest |
| `manifest_provenance --check` | both digests recompute |
| guard test, and its inert-proof | both pass |

## 7. Found and left alone

Per the standing rule that a second change riding along is unattributable
later, these are recorded, not fixed:

- **`summary_<tag>.json` and `report_<tag>.txt` still default under `data/`.**
  They are the same class of output as the manifest — counts, timings and
  `missing_reason` tallies, no series values — and by the argument in §4 they
  could be committed too. They are also fully derivable from the manifest and
  the store, which is why they were not moved with it. Nothing points at them
  from a committed document.
- ~~**Four analysis entry points still name a manifest by a path of their
  own.**~~ **Closed by docs/P3_REPO_HYGIENE.md §2.** `benchmarks.loss_tables`,
  `benchmarks.defect_tables` and `benchmarks.convergence_forensics` defaulted
  `--manifest` to the gitignored `data/grid_primary/manifest_fix.json`, so a
  clean checkout could not run them even though the file they wanted was
  committed a directory away; all three now read `docs/P3_GRID_manifest.json`,
  and re-running `loss_tables` reproduced its four committed outputs
  byte-for-byte. `benchmarks.tsfm_distribution_probe` already defaulted to the
  committed path, so its default changed meaning when that file was promoted —
  from the pre-fix grid to the post-fix one — and
  docs/P3_TSFM_VARIANCE_AUDIT.md now says at the top which grid it was
  measured on.
- **`docs/P3_CONVERGENCE_FITS.parquet` is untracked**, caught by `.gitignore`'s
  blanket `*.parquet` rather than by the root-anchored `/data/` rule. It is a
  committed document's evidence file that cannot currently be committed. This
  is the same disease in a different organ and wants its own decision: whether
  the `*.parquet` rule should be narrowed, or that evidence re-expressed.
- **`docs/P3_GRID.md` still names `data/grid_primary/run_grid.py`** as the
  driver, and now also predates the manifest's move. It is the run report for a
  specific run and is left as the record it is, exactly as
  docs/P3_DRIVER_PROVENANCE.md §7 left it.
- **`docs/decisions.md` in this mirror stops at D-032** and does not contain
  D-042, which prompt M cites for the principle that a check which cannot fail
  is not a check. Flagged as mirror drift for the planning machine; nothing was
  edited here.
