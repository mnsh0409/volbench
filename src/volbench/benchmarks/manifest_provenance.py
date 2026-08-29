#!/usr/bin/env python
"""One current grid manifest, committed, self-describing and recomputable.

The run manifest is the only thing that says which fragments in an append-only
store are the current ones. After the L fix run the store holds 187 fragments —
the 143 the study scores and the 44 those replaced — and telling them apart
needs the manifest. It was written under ``data/``, which is gitignored, so a
clean checkout could not tell; ``docs/P3_GRID_manifest.json`` was committed but
named the pre-fix hashes for the 44 replaced cells.

That was the third instance of one pattern (the driver in
docs/P3_DRIVER_PROVENANCE.md, the input digests in
``benchmarks.data_digests``), so this module is the general fix rather than
another one-off:

* :data:`CURRENT_MANIFEST` is *always* the current grid manifest, and
  ``benchmarks.grid_primary`` writes its manifest under ``docs/`` by default.
  Only the store — the parquet fragments and their sidecars, which carry values
  derived from licensed market data — stays under ``data/``. A manifest carries
  hashes, versions, environment pins, timings and per-cell status, and no
  series value; that is what lets it be committed at all, and
  ``tests/test_manifest_provenance.py`` checks it field by field rather than
  taking it on trust.
* Superseded manifests are archived verbatim under :data:`ARCHIVE_DIR` rather
  than deleted: J1's evidence documents are anchored to the pre-fix grid and
  have to stay checkable.
* :func:`orphaned_manifests` is the guard behind
  ``tests/test_manifest_provenance.py``: no manifest may exist under ``data/``
  without a committed counterpart under ``docs/``.

**Provenance here is anchored on content digests, not commit SHAs.** The public
release repository will not carry this development history, so a manifest that
names its own digest is citable from the paper and verifiable by recomputation
where a commit SHA is not. Two recipes, both implemented below and both
reproducible from the committed files alone:

``manifest_digest``
    SHA-256 of the manifest with the three provenance fields
    (:data:`PROVENANCE_FIELDS`) removed, serialised as UTF-8 JSON with sorted
    keys and no whitespace (``separators=(",", ":")``). Excluding those three
    is what makes the digest a statement about *the run* rather than about the
    envelope, so an annotated manifest and the bare one the driver wrote share
    it — which is exactly the counterpart relation the guard needs.

``store_digest``
    SHA-256 of one line per cell, ``"{config_hash}  {parquet}  {sidecar}"``
    with the two file SHA-256s, sorted by config hash, joined by ``\\n`` with a
    trailing newline. It binds the manifest both to *which* fragments it names
    and to their bytes.

Verify a checkout against its own manifest, or promote a run's manifest into
the committed one::

    uv run python -m volbench.benchmarks.manifest_provenance --check
    uv run python -m volbench.benchmarks.manifest_provenance \\
        --promote data/grid_primary/manifest_fix.json

``--check`` recomputes both digests and exits non-zero on the first
disagreement. ``--promote`` archives whatever it replaces, records that file's
digest and archived path in ``supersedes``, and writes the annotated manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

__all__ = [
    "ARCHIVE_DIR",
    "CURRENT_MANIFEST",
    "DEFAULT_STORE",
    "IDENTIFYING_FIELDS",
    "PROVENANCE_FIELDS",
    "annotate",
    "archive",
    "core_digests",
    "find_manifests",
    "fragment_lines",
    "manifest_digest",
    "orphaned_manifests",
    "problems",
    "redact",
    "store_digest",
    "supersede",
]

#: The current grid manifest. There is exactly one, it is committed, and it is
#: this path — the whole point of this module.
CURRENT_MANIFEST: Final = Path("docs/P3_GRID_manifest.json")

#: Superseded manifests and resume-verification artifacts, kept verbatim. The
#: directory name is the statement that nothing in it is current.
ARCHIVE_DIR: Final = Path("docs/archive")

#: The fragment store the manifest names. Stays under ``data/`` forever: it
#: holds values derived from Stooq and Binance data (docs/data_licenses.md).
DEFAULT_STORE: Final = Path("data/grid_primary/store")

#: The envelope: what the manifest says about itself rather than about the run.
#: Excluded from :func:`manifest_digest`.
PROVENANCE_FIELDS: Final = ("manifest_digest", "store_digest", "supersedes")

#: Fields removed from a manifest before it is published, by path from the
#: root. They identify the person who ran the study rather than the run: IJF
#: review is double-blind and the reproducibility package is part of what a
#: reviewer sees (``tests/test_identity_leakage.py``). ``interpreter.executable``
#: was an absolute path under a home directory; the interpreter *version* stayed,
#: because the version is what reproduces a run and the venv path is meaningful
#: only on the machine that already has it.
#:
#: ``determinism.interpreter_info`` no longer produces it, so this is a
#: migration for manifests written before that changed, not a filter new runs
#: rely on.
IDENTIFYING_FIELDS: Final = (("environment", "interpreter", "executable"),)


def _canonical(payload: Any) -> bytes:
    """The one serialisation both digests are computed over."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """SHA-256 of the manifest's run content, provenance envelope excluded.

    Two manifests share this digest exactly when they make the same statement
    about the same run, whether or not one of them has been annotated.
    """
    core = {key: value for key, value in manifest.items() if key not in PROVENANCE_FIELDS}
    return hashlib.sha256(_canonical(core)).hexdigest()


def fragment_lines(manifest: Mapping[str, Any], store_root: Path) -> list[str]:
    """``"{config_hash}  {parquet sha256}  {sidecar sha256}"``, sorted.

    Raises if a named fragment is missing: a manifest whose cells do not
    resolve is not a manifest of anything that is on this machine, and a
    digest computed over the ones that happen to be there would hide that.
    """
    hashes = sorted({c["config_hash"] for c in manifest["cells"] if c["config_hash"]})
    lines = []
    for config_hash in hashes:
        parquet = store_root / f"{config_hash}.parquet"
        sidecar = store_root / f"{config_hash}.json"
        missing = [p.name for p in (parquet, sidecar) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{store_root}: manifest names {config_hash}, missing {missing}"
            )
        lines.append(f"{config_hash}  {_file_sha256(parquet)}  {_file_sha256(sidecar)}")
    return lines


def store_digest(manifest: Mapping[str, Any], store_root: Path = DEFAULT_STORE) -> str:
    """SHA-256 of the fragment set the manifest names, by identity and bytes."""
    body = "\n".join(fragment_lines(manifest, store_root)) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def annotate(
    manifest: Mapping[str, Any],
    *,
    store_root: Path = DEFAULT_STORE,
    supersedes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The manifest with its three provenance fields, envelope first.

    Idempotent: any envelope already present is recomputed, never carried over,
    so re-annotating a file cannot preserve a digest that has stopped being
    true.
    """
    core = {key: value for key, value in manifest.items() if key not in PROVENANCE_FIELDS}
    envelope: dict[str, Any] = {
        "manifest_digest": manifest_digest(core),
        "store_digest": store_digest(core, store_root),
        "supersedes": dict(supersedes) if supersedes is not None else None,
    }
    return {**envelope, **core}


def archive(path: Path, archive_dir: Path = ARCHIVE_DIR) -> Path:
    """Copy *path* into the archive verbatim, under a content-addressed name.

    Verbatim and content-addressed on purpose: the archived bytes are the bytes
    the run wrote, so an archived manifest keeps the same ``manifest_digest``
    as the file it came from, and archiving the same file twice is a no-op
    rather than a second copy under a second name.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{path.stem}.{_file_sha256(path)[:12]}.json"
    if not destination.exists():
        shutil.copyfile(path, destination)
    return destination


def redact(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest with :data:`IDENTIFYING_FIELDS` removed, deeply.

    Changes ``manifest_digest``, deliberately and once: the digest is the run's
    citable anchor, so a redacted manifest is a different — publishable —
    statement about the same run, and every document citing the old digest has
    to move with it. Applied to *every* copy of a manifest rather than only the
    committed one, so the counterpart relation :func:`orphaned_manifests`
    depends on keeps holding.
    """
    out = json.loads(json.dumps(manifest))  # a deep copy that cannot alias
    for path in IDENTIFYING_FIELDS:
        node = out
        for key in path[:-1]:
            if not isinstance(node, dict) or key not in node:
                break
            node = node[key]
        else:
            if isinstance(node, dict):
                node.pop(path[-1], None)
    return dict(out)


def supersede(path: Path, archive_dir: Path = ARCHIVE_DIR) -> dict[str, Any] | None:
    """Archive the manifest at *path*, and describe it for a ``supersedes`` field.

    ``None`` when there is nothing there — the first run of a fresh checkout
    supersedes nothing, and saying so is more useful than an absent field.
    """
    if not path.exists():
        return None
    archived = archive(path, archive_dir)
    return {
        "manifest_digest": manifest_digest(_load(path)),
        "file_sha256": _file_sha256(path),
        "archived_path": str(archived),
    }


def find_manifests(root: Path) -> list[Path]:
    """Every run manifest under *root*, at any depth.

    Matched by name *and* shape: the store's per-fragment sidecars and the
    runs' ``summary_*.json`` live in the same trees and are not manifests.
    """
    found = []
    for path in sorted(root.rglob("*manifest*.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and "cells" in payload and "n_cells" in payload:
            found.append(path)
    return found


def core_digests(paths: Iterable[Path]) -> dict[str, Path]:
    """``manifest_digest -> path`` for readable manifests among *paths*."""
    digests: dict[str, Path] = {}
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and "cells" in payload:
            digests.setdefault(manifest_digest(payload), Path(path))
    return digests


def orphaned_manifests(data_root: Path, committed: Mapping[str, Path]) -> list[Path]:
    """Manifests under *data_root* that no committed manifest accounts for.

    *committed* is ``manifest_digest -> path`` over the manifests tracked under
    ``docs/``. The relation is the run digest rather than file identity, so a
    committed manifest still accounts for the bare file the driver wrote after
    it has been annotated with its provenance envelope.
    """
    if not data_root.exists():
        return []
    return [p for p in find_manifests(data_root) if manifest_digest(_load(p)) not in committed]


def _load(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def problems(path: Path = CURRENT_MANIFEST, store_root: Path = DEFAULT_STORE) -> list[str]:
    """Every disagreement between a manifest, its own digests and the store."""
    manifest = _load(path)
    found: list[str] = []
    for field in PROVENANCE_FIELDS:
        if field not in manifest:
            found.append(f"{path}: no {field} field — it is not self-describing")
    if "manifest_digest" in manifest:
        recomputed = manifest_digest(manifest)
        if recomputed != manifest["manifest_digest"]:
            found.append(
                f"{path}.manifest_digest: recorded {manifest['manifest_digest']!r} != "
                f"recomputed {recomputed!r}"
            )
    if "store_digest" in manifest:
        try:
            recomputed = store_digest(manifest, store_root)
        except FileNotFoundError as exc:
            found.append(str(exc))
        else:
            if recomputed != manifest["store_digest"]:
                found.append(
                    f"{path}.store_digest: recorded {manifest['store_digest']!r} != "
                    f"recomputed {recomputed!r} over {store_root}"
                )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CURRENT_MANIFEST)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute the committed manifest's own digests against the store",
    )
    parser.add_argument(
        "--redact",
        type=Path,
        nargs="+",
        default=None,
        metavar="MANIFEST",
        help="strip the identifying fields from these manifests in place, and stop",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        nargs="+",
        default=None,
        metavar="MANIFEST",
        help="copy these manifests into the archive verbatim, and stop",
    )
    parser.add_argument(
        "--promote",
        type=Path,
        default=None,
        metavar="RUN_MANIFEST",
        help="archive the current manifest and install this one in its place",
    )
    args = parser.parse_args(argv)

    if args.redact is not None:
        for target in args.redact:
            before = _load(target)
            after = redact(before)
            if after == before:
                print(f"unchanged  {target}")
                continue
            if any(field in after for field in PROVENANCE_FIELDS):
                # The envelope described the un-redacted file; recompute it
                # rather than leave a digest that no longer verifies.
                after = annotate(
                    after, store_root=args.store, supersedes=after.get("supersedes")
                )
            target.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
            print(f"redacted   {target}  -> manifest_digest {manifest_digest(after)}")
        return 0

    if args.archive is not None:
        for source in args.archive:
            print(f"archived {source} -> {archive(source, args.archive_dir)}")
        return 0

    if args.promote is not None:
        superseded = supersede(args.manifest, args.archive_dir)
        if superseded is not None:
            print(f"archived {args.manifest} -> {superseded['archived_path']}")
        annotated = annotate(_load(args.promote), store_root=args.store, supersedes=superseded)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(annotated, indent=2) + "\n", encoding="utf-8")
        print(f"promoted {args.promote} -> {args.manifest}")
        print(f"  manifest_digest {annotated['manifest_digest']}")
        print(f"  store_digest    {annotated['store_digest']}")
        return 0

    if not args.check:
        parser.error("nothing to do: pass --check, --promote, --archive or --redact")

    found = problems(args.manifest, args.store)
    if found:
        print(f"MANIFEST PROVENANCE MISMATCH ({len(found)}):")
        for line in found:
            print(f"  {line}")
        return 1
    manifest = _load(args.manifest)
    print(
        f"{args.manifest}: {manifest['n_cells']} cells, digests recompute "
        f"(manifest {manifest['manifest_digest'][:12]}, store {manifest['store_digest'][:12]})"
    )
    return 0


if __name__ == "__main__":
    sys.argv[0] = "manifest_provenance"
    raise SystemExit(main())
