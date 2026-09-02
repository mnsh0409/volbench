# Prompt R — Data provenance diagnostic

**READ-ONLY on `src/`. No fixes, no reruns, no network.**

## Context

You are in the volbench repo on the 4090. On **2026-08-31 at 02:54:27** local time, nine
parquet files appeared in `data/cache/stooq/` under a new date key
(`stooq_<symbol>_2026-08-30.parquet`), while an unattended three-arm ablation chain was
running (`run/arms_overnight.sh`, started 2026-08-30 23:05, ended 2026-08-31 03:57,
`exit=0`). Nobody was at the machine. No cron entry and no systemd timer accounts for it —
the user's crontab contains only CHI-2027 and GO/GPU jobs, and `systemctl --user
list-timers` is empty.

Two things must be established. Neither is currently known, and this session must not
guess at either.

**1. Mechanism.** `src/volbench/data/stooq.py` contains *both*:

- a live HTTP fetch to `https://stooq.com/q/d/l/` — `import requests` (line 65),
  `fetch_stooq_csv` (line 126), called from the cache-writing function at line 276; and
- a second cache-writing path around lines 318–337 whose docstring says the data was
  "ingested" and that `parse_stooq_csv` "reads both layouts" — i.e. a manual-download
  ingest.

Both construct the same filename via the stem at line 245
(`stooq_<symbol>_<download_date>.parquet`) and both default `cache_dir` to
`data/cache/stooq`. Which one ran on 2026-08-31 is unknown. Nine files written inside
0.18 s argues against nine sequential round-trips to a European host, but does not rule
out concurrent fetches with batched writes.

**2. Attribution.** Which process invoked it, and whether any experiment's results depend
on the resulting data change. Two competing hypotheses, both live:

- **(A)** the ablation chain itself triggered it (arm 3 has three target legs; the third
  appears to start around 02:56 by directory mtime);
- **(B)** something outside volbench triggered it — the user has a separate CHI-2027
  project on this machine with its own crawler.

There is evidence against a lazy per-experiment load: the cache batches cover **ten**
symbols (`idx_ndx`, `idx_dax`, `idx_cac`, `idx_nkx`, `idx_hsi`, `idx_twse`, `idx_kospi`,
`SPY`, `DIA`, `ISF`) while the benchmark grid uses far fewer. A whole-panel refresh is not
what a single experiment leg loading its own assets would produce. Test both hypotheses;
do not assume either.

## Hard constraints

- **Make no network request to any host.** The one probe that exercises the loader blocks
  sockets *first* (Task 2). If any step would contact stooq.com, stop and report instead.
- **Do not modify anything under `src/`.** This session is read-only on the source tree.
- **Do not delete, move, or overwrite anything under `data/`.** Point probes at a temp
  directory.
- Do not rerun any experiment, and do not switch branches.
- **The report must contain no individual market observation** — no price, no return, no
  high, no low, no single realised value. Counts, dates, and aggregate magnitudes only.
  The licensing guard and `COLUMN_POLICY` bind anything you commit.

Governing rule for this repo, stated so you do not have to infer it: **Stooq data must
never be fetched programmatically.** The site blocks automated access by robots.txt and by
deliberate anti-bot policy; manual browser download is the only permitted route, and that
is permanent. If a live fetch path turns out to be reachable from the benchmark pipeline,
that is a defect to **report here** and fix in a later session — not to fix in this one.

## Task 1 — Read the two cache-writing paths and their call chain

Read `src/volbench/data/stooq.py` in full, then report:

1. The exact signature of both cache-writing functions (the one calling `fetch_stooq_csv`
   around lines 245–290, and the one around lines 310–345).
2. **The default value of `download_date` in each.** If it is `date.today()` or similar,
   say so explicitly — that is the mechanism by which a cache key rolls over unattended.
3. Whether each function short-circuits on an existing cache file *before* reaching the
   fetch, and at which line. Quote the branch.
4. Where the second path reads its bytes from — a directory of manually downloaded CSVs, a
   parameter, stdin? Give the path if it is a default.
5. Every call site of both functions across the repo:
   `grep -rn "<name1>\|<name2>" --include='*.py' src/ scripts/ tests/`
6. **Trace the call chain from the grid driver down to the cache write.** Start at
   `volbench.benchmarks.grid_primary` and follow it. State plainly which of the two
   functions the grid can reach, or that it reaches neither. If it reaches neither, say
   what the grid does load and from where.
7. Whether anything in the repo performs a **whole-panel refresh** across all ten symbols
   in one call, and if so, what invokes it.

## Task 2 — Settle fetch-vs-ingest offline

Write a throwaway script under `/tmp` (**not** in the repo). It must, in this order:

1. Block the network before importing volbench:

```python
import socket
class NetBlocked(RuntimeError): pass
def _blocked(*a, **k): raise NetBlocked("network blocked by diagnostic probe")
socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
```

2. Import the loader and call the path the grid reaches (from Task 1.6), with
   `cache_dir=Path("/tmp/volbench_probe_cache")` and a `download_date` that has no cache
   entry (e.g. `date(2026, 9, 9)`), for **one** symbol.

Report which happened:

- **Raises `NetBlocked`** (directly, or wrapped in a `requests` connection error) → the
  pipeline reaches the fetch. Report the traceback's frame list so the call chain is on the
  record.
- **Succeeds and writes a parquet under `/tmp/volbench_probe_cache`** → the pipeline
  re-derives locally. Report exactly which file(s) on disk it read to do so.
- **Raises something else** → report it verbatim; do not interpret.

Then repeat with a `download_date` that *does* have a cache entry, to confirm the
short-circuit works. Nothing may be written under `data/`; verify that and say so.

## Task 3 — Are the two cache vintages actually different?

Both `stooq_<symbol>_2026-08-29.parquet` and `stooq_<symbol>_2026-08-30.parquet` are on
disk. For each of the nine symbols present in both, report:

- row count in each, and the delta;
- first and last date in each;
- whether the **overlapping** date range is exactly equal across all columns;
- if not exactly equal: how many rows differ, on which dates, and the order of magnitude of
  the largest absolute difference (e.g. "< 1e-12", "~1e-4", "> 0.01"). **No raw values.**

This is the question that decides whether anything downstream actually moved. Equal on the
overlap with only trailing rows added is a very different finding from restated history.

## Task 4 — Which snapshot did each store consume?

For every store directory under `data/` (`grid_primary`, the three `grid_ablation_*`, the
three `leakage_canary*`, `toy_benchmark`, `serial_parallel_gate`, `lgbm_smear_probe`,
`fit_probe`, `tsfm_dist_probe`, `smoke_tsfm`, `panel`):

1. Report the run window as min and max fragment mtime.
2. Report which `download_date` cache keys existed during that window, and which was newest.
3. Flag any store whose run window **straddles** a cache-key creation time.

Then, specifically for `data/grid_ablation_targets/store`: **group its 351 fragments by
target and report min/max mtime per target.** This establishes the real leg boundaries from
the fragments themselves rather than from directory mtimes, and shows whether any leg
straddles 02:54:27. Name which targets fall on which side.

## Task 5 — Attribution: was it volbench or something else?

1. `grep -n` the arms log for every line within ±5 minutes of `02:54:27`, and report the
   log's own timestamps for each leg boundary. Quote the surrounding lines.
2. `ps -eo pid,lstart,etime,cmd --sort=lstart` — report any long-running process whose
   command line mentions volbench, stooq, python, or a data refresh.
3. Test hypothesis (B) directly:
   `grep -rIn "volbench\|stooq" "$HOME/Documents/CHI/2027/chi2027_kit" /mnt/nvme2/research/chi2027 2>/dev/null | head -50`
   Report whether anything outside this repo imports or invokes volbench's data layer.
4. Check `~/.bash_history` and any shell history file for invocations near the batch times
   (2026-08-25 04:38, 08-25 14:01, 08-26 07:34, 08-26 15:38, 08-29 00:25, 08-29 23:07,
   08-31 02:54). Report commands only; note that history may be untimestamped.
5. State which hypothesis the evidence supports, **and what would still refute it.** If the
   evidence is not decisive, say so — do not pick a side to be helpful.

## Task 6 — Does the manifest record the input data at all?

Read one manifest JSON under `docs/` (e.g. `P3_GRID_manifest_target_squared_return.json`)
and list its **top-level keys**. Report whether any field identifies the input data — a
snapshot id, an input-file list, a content digest of the loaded series, anything. State
plainly whether `manifest_digest` and `store_digest` can distinguish two runs that consumed
different input vintages.

## Deliverable

Write `docs/P3_DATA_PROVENANCE.md` containing all six task outputs, with a two-paragraph
summary at the top answering: (i) does the benchmark pipeline reach a live Stooq fetch,
yes or no; (ii) did any committed result consume more than one data vintage.

Commit on a new branch `diag/data-provenance` and push. Source tree unchanged — `git
status` must show no modifications under `src/`. Confirm the licensing guard and identity
guard still pass before committing.

End with: what you could not determine, and what would determine it.
