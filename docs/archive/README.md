# Superseded manifests — none of these is current

`docs/P3_GRID_manifest.json` is always the current grid manifest. Everything in
this directory is a manifest that *was* current, or the record of a verification
run that recomputed nothing. They are kept because deleting them would break
things that point at them: J1's leakage-canary and determinism evidence is
anchored to the pre-fix grid, and a superseded manifest is the only thing that
says which of the store's 187 fragments that evidence was about.

Files are named `<the name the run wrote>.<first 12 hex of the file's SHA-256>.json`
and are byte-identical copies. Content-addressing means archiving the same file
twice is a no-op rather than a second copy under a second name, and it means an
archived manifest keeps the `manifest_digest` of the file it came from — which
is the relation `tests/test_manifest_provenance.py` uses to decide that a
manifest under gitignored `data/` is accounted for.

Add to this directory through the tool, never by hand:

    uv run python -m volbench.benchmarks.manifest_provenance --archive <path>

See `docs/P3_MANIFEST_INVENTORY.md` for what each file is, when it was written,
and which fragment set it names.
