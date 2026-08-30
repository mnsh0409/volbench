# Prompt P — Expose the protocol settings the ablation arms need, then hand back the launch commands

**Branch:** `git switch feat/p3-analysis && git pull`, then `git switch -c feat/ablation-arms`.

**Session:** fresh — `claude -n vb-p-ablations`. Run `/effort high`.

**Two halves, and you only do the first.** Part A is a code change with verification. Part B is hours of unattended compute — **do not start it.** Report the exact commands and let them be launched separately, so the machine is not tied to a session overnight.

---

## Why this exists

D-034 schedules three ablation arms before the results freeze, and `--help` shows the driver cannot express any of them: the protocol is fixed at horizon 1, window 500, step 1, `refit_every` 21, daily re-conditioning, per-asset primary target.

The three arms:

1. **Window 1000** (D-019's robustness arm against the 500 primary)
2. **`recondition="none"`** — D-011's efficiency reading, and the control for D-015's "re-estimate every N, re-condition daily" claim
3. **Robustness targets** — Parkinson, Garman–Klass and squared close-to-close on the **nine equity assets only** (D-004; crypto keeps 5-minute realized variance, which has no range-estimator analogue)

## Part A — expose them properly

Add `--window`, `--recondition` and `--target` (or whatever the existing arm object calls them — match its vocabulary, do not invent new names).

**The one rule that matters: they must feed the existing arm/config plumbing, not a parallel path.** Every one of these settings already reaches the config hash today; a flag that sets them anywhere else would produce fragments whose hash does not describe them, which is exactly the failure D-032 spent a day diagnosing.

Verify rather than assume:

- **A test per flag asserting the config hash moves** when the flag changes, and that it does not move when the flag is set to its current default. Both halves — a flag that always changes the hash is as broken as one that never does.
- Confirm the defaults reproduce the primary grid: run one cheap cell (say SPY `garch11`) with no new flags and assert its hash equals the one already in `docs/P3_GRID_manifest.json`.

## Part B — establish, don't run

For each arm, **measure and report**, without launching the full run:

1. **Which cells' hashes actually move.** Window enters every cell's hash, so arm 1 shares nothing with the primary store. `recondition` enters the hash only when `refit_every > 1` and only for models with an update path, so arm 2 may share most of its cells. Target is per-cell, so arm 3 moves all 117 equity cells. **Count them.**
2. **Therefore, shared store or separate `--out-dir`.** Where an arm shares nothing, a separate store keeps the primary's closure property (M's 143 + 44 = 187) intact and makes the arm independently deletable. Where an arm shares most cells, the same store avoids recomputing a 68-minute GPU lane for nothing. Decide per arm on the counts, and say why.
3. **A smoke run per arm** — one asset, the cheap CPU models, its own tag — proving the arm runs end to end and its hashes differ from the primary as expected. Minutes, not hours.
4. **Wall-clock estimate** for each full arm, extrapolated from the smoke run. Note that window 1000 is *not* simply the primary's runtime: fewer origins, but longer fits and longer TSFM context.
5. **Disk.** Report free space and the projected fragment volume. A previous session found root at 100% with 4.7 G free.

Every arm uses **its own `--tag`**, so each writes a sibling manifest under `docs/` and none touches `docs/P3_GRID_manifest.json`. Arm 3 is restricted with `--assets`, which the driver refuses under the default tag anyway.

## Then hand back the commands

Report, ready to paste, the exact `nohup` line for each arm — with the Makefile's determinism exports, the right `uv run` extras, the tag, the out-dir you chose, and the log path. Verified, not guessed: you will have run each shape already in the smoke test.

## Carry this forward into whatever analyses these arms

D-019: **the arms do not score the same origins.** A window-1000 cell has 500 fewer origins than its window-500 counterpart, so any window-sensitivity comparison must run on the **intersection of origins scored by both**, or be reported per arm with explicit coverage. Put that in the arm's own documentation now, while it is obvious, rather than leaving it for whoever compares them later.

## Guards, as usual

`ruff`, `mypy --strict`, full suite on all three interpreters. `git ls-files -- data/` empty. The licensing guard, the identity-leakage test, M's manifest counterpart guard and O's column policy all green. A resumability re-run of the **primary** grid still reporting 143 cached, 0 computed, with every fragment byte-identical and unrewritten — the point being that adding these flags disturbed nothing. Push the branch; do not merge.

Report what you changed, the hash tests and their two halves, the per-arm cell counts and store decision, the smoke results, the estimates, and the launch commands.
