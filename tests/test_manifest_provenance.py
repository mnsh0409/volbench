"""One current grid manifest, committed, and no manifest stranded under ``data/``.

The store is append-only: after the L fix run it holds 187 fragments, the 143
the study scores and the 44 those replaced. Only the manifest says which set is
current, so a manifest that exists only under gitignored ``data/`` makes a
clean checkout unable to tell — which is what happened, for the third time in
this phase (docs/P3_MANIFEST_INVENTORY.md). These tests are the mechanical half
of the fix.

Two of them need ``data/``, which is neither in a clean checkout nor in CI, and
skip without it. The rest do not: the manifest is committed, so its shape, its
self-consistency and the guard's ability to *fail* are all checkable anywhere —
and the last of those is the one that matters. A guard that passes because
there is nothing under ``data/`` to check is not a guard, so it is proved
against a planted manifest rather than trusted.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from volbench.benchmarks.manifest_provenance import (
    ARCHIVE_DIR,
    CURRENT_MANIFEST,
    DEFAULT_STORE,
    PROVENANCE_FIELDS,
    core_digests,
    find_manifests,
    manifest_digest,
    orphaned_manifests,
    problems,
    store_digest,
)

REPO = Path(__file__).parents[1]
SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: Every key a cell may carry. The manifest is committed, and the argument that
#: lets it be is that it holds no series values — so the keys are enumerated
#: rather than trusted, the same way docs/P3_DRIVER_PROVENANCE.md §3 enumerated
#: the driver's literals.
CELL_KEYS = {
    "index", "asset", "model", "horizon", "arm", "lane", "status", "config_hash",
    "n_rows", "n_missing", "n_fits", "n_fits_fallback", "n_fits_nonconverged",
    "wall_clock_s", "error",
}

#: The only cell key whose value is a float. Everything else is a count, an
#: index, a label or a hash — nothing that could be a price, a return or a
#: variance.
FLOAT_CELL_KEYS = {"wall_clock_s"}


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


def _committed_docs_manifests() -> dict[str, Path]:
    """``manifest_digest -> path`` over the manifests git tracks under ``docs/``."""
    tracked = [
        REPO / line
        for line in _git("ls-files", "--", "docs/").stdout.split()
        if "manifest" in Path(line).name.lower() and line.endswith(".json")
    ]
    return core_digests(tracked)


needs_git = pytest.mark.skipif(
    shutil.which("git") is None or _git("rev-parse", "--git-dir").returncode != 0,
    reason="not a git checkout",
)
needs_store = pytest.mark.skipif(
    not (REPO / DEFAULT_STORE).exists(),
    reason="no local store: the fragments are gitignored and absent in CI",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(
        (REPO / CURRENT_MANIFEST).read_text(encoding="utf-8")
    )
    return payload


class TestTheCommittedManifest:
    def test_it_is_self_describing(self, manifest: dict[str, Any]) -> None:
        """The three fields that make it citable without a commit SHA. The
        release repository will not carry this history, so provenance is
        anchored on content digests instead."""
        for field in PROVENANCE_FIELDS:
            assert field in manifest, f"{CURRENT_MANIFEST} has no {field}"
        assert SHA256.match(manifest["manifest_digest"])
        assert SHA256.match(manifest["store_digest"])

    def test_its_own_digest_recomputes(self, manifest: dict[str, Any]) -> None:
        """``manifest_digest`` is over the run content, envelope excluded — so
        it survives re-annotation and is what a reader can recompute."""
        assert manifest_digest(manifest) == manifest["manifest_digest"]

    def test_it_covers_the_whole_grid_once(self, manifest: dict[str, Any]) -> None:
        cells = manifest["cells"]
        assert manifest["n_cells"] == len(cells) == 143
        assert len({c["config_hash"] for c in cells}) == 143
        assert manifest["n_failed"] == 0
        assert manifest["n_computed"] + manifest["n_cached"] == 143

    def test_it_records_the_interpreter(self, manifest: dict[str, Any]) -> None:
        """J1 had to establish CPython 3.11.5 three indirect ways because the
        run recorded it nowhere. It records it now."""
        interpreter = manifest["environment"]["interpreter"]
        assert interpreter["implementation"] and interpreter["python"]

    def test_it_records_the_determinism_pins(self, manifest: dict[str, Any]) -> None:
        environment = manifest["environment"]
        assert environment["blas_threads"] == 1  # D-032
        assert environment["env"]["NPY_DISABLE_CPU_FEATURES"]  # D-026
        assert environment["kernel_signature"]

    def test_it_contains_no_series_values(self, manifest: dict[str, Any]) -> None:
        """The licensing guard forbids tracking anything under ``data/``; this
        file is under ``docs/`` and stays publishable because a hash, a count
        and a wall clock are not the series. Enumerated, not assumed."""
        for cell in manifest["cells"]:
            assert set(cell) == CELL_KEYS
            assert SHA256.match(cell["config_hash"])
            for key, value in cell.items():
                if isinstance(value, float):
                    assert key in FLOAT_CELL_KEYS, f"unexpected float {key}={value}"

    def test_it_names_the_manifest_it_replaced(self, manifest: dict[str, Any]) -> None:
        """Superseded manifests are archived, not deleted: J1's evidence
        documents are anchored to the pre-fix grid and stay checkable."""
        superseded = manifest["supersedes"]
        assert superseded is not None
        archived = REPO / superseded["archived_path"]
        assert archived.is_file(), f"{archived} is named but absent"
        assert ARCHIVE_DIR in archived.relative_to(REPO).parents
        previous = json.loads(archived.read_text(encoding="utf-8"))
        assert manifest_digest(previous) == superseded["manifest_digest"]
        assert manifest_digest(previous) != manifest["manifest_digest"]

    @needs_store
    def test_its_digests_agree_with_the_store_on_this_machine(self) -> None:
        assert problems(REPO / CURRENT_MANIFEST, REPO / DEFAULT_STORE) == []


class TestNoManifestIsStrandedUnderData:
    """The guard, and the proof that it is not decoration.

    ``tests/test_licensing_guard.py::TestNoDataIsTracked`` answers a different
    question — is anything under ``data/`` *tracked* — and is unchanged. This
    one asks whether anything under ``data/`` is the only copy of itself.
    """

    @needs_git
    @needs_store
    def test_the_repository_has_none(self) -> None:
        orphans = orphaned_manifests(REPO / "data", _committed_docs_manifests())
        assert orphans == [], (
            "manifests exist only under gitignored data/: "
            f"{[str(p.relative_to(REPO)) for p in orphans]}. Archive them with "
            "`python -m volbench.benchmarks.manifest_provenance --archive <path>`."
        )

    def test_the_guard_fires_on_a_planted_manifest(self, tmp_path: Path) -> None:
        """The inert-proof. Once the driver writes to ``docs/`` there may be no
        manifest under ``data/`` at all, and a check that passes because there
        is nothing to check reports "verified" either way."""
        data_root = tmp_path / "data" / "grid_primary"
        data_root.mkdir(parents=True)
        planted = data_root / "manifest_stranded.json"
        planted.write_text(json.dumps({"n_cells": 1, "cells": [{"config_hash": "ab" * 32}]}))

        assert orphaned_manifests(data_root, {}) == [planted]

    def test_a_committed_counterpart_clears_it(self, tmp_path: Path) -> None:
        """The other half: the guard must not fire on a manifest that *is*
        accounted for, or it would be noise and get switched off."""
        data_root = tmp_path / "data"
        data_root.mkdir()
        payload = {"n_cells": 1, "cells": [{"config_hash": "ab" * 32}]}
        (data_root / "manifest_run.json").write_text(json.dumps(payload))

        assert orphaned_manifests(data_root, {manifest_digest(payload): Path("docs/x.json")}) == []

    def test_annotation_does_not_break_the_counterpart_relation(self, tmp_path: Path) -> None:
        """The committed manifest carries three fields the driver's file does
        not, so the relation is the run digest, never byte identity — otherwise
        promoting a manifest would immediately strand the file it came from."""
        data_root = tmp_path / "data"
        data_root.mkdir()
        payload = {"n_cells": 1, "cells": [{"config_hash": "ab" * 32}]}
        (data_root / "manifest_run.json").write_text(json.dumps(payload))
        committed = {**payload, "manifest_digest": "x", "store_digest": "y", "supersedes": None}

        assert orphaned_manifests(data_root, core_digests([_write(tmp_path, committed)])) == []

    def test_it_looks_at_manifests_and_not_at_the_store(self, tmp_path: Path) -> None:
        """The store's per-fragment sidecars and the runs' summaries live in
        the same tree and are not manifests; matching them would make the guard
        unsatisfiable."""
        data_root = tmp_path / "data"
        (data_root / "store").mkdir(parents=True)
        (data_root / "store" / f"{'ab' * 32}.json").write_text('{"model": {}, "data": {}}')
        (data_root / "summary_primary.json").write_text('{"tag": "primary"}')
        (data_root / "manifest_notes.json").write_text('{"note": "not a run manifest"}')

        assert find_manifests(data_root) == []


class TestTheDigestRecipes:
    def test_the_envelope_is_outside_the_manifest_digest(self) -> None:
        payload: dict[str, Any] = {"n_cells": 0, "cells": []}
        annotated = {**payload, "manifest_digest": "x", "store_digest": "y", "supersedes": {}}
        assert manifest_digest(annotated) == manifest_digest(payload)

    def test_the_run_content_is_inside_it(self) -> None:
        payload: dict[str, Any] = {"n_cells": 1, "cells": [{"config_hash": "ab" * 32}]}
        moved = {"n_cells": 1, "cells": [{"config_hash": "cd" * 32}]}
        assert manifest_digest(payload) != manifest_digest(moved)

    def test_the_store_digest_follows_the_fragment_bytes(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        config_hash = "ab" * 32
        (store / f"{config_hash}.parquet").write_bytes(b"one")
        (store / f"{config_hash}.json").write_bytes(b"{}")
        payload = {"n_cells": 1, "cells": [{"config_hash": config_hash}]}

        before = store_digest(payload, store)
        (store / f"{config_hash}.parquet").write_bytes(b"two")
        assert store_digest(payload, store) != before

    def test_a_missing_fragment_is_an_error_not_a_shorter_digest(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        payload = {"n_cells": 1, "cells": [{"config_hash": "ab" * 32}]}
        with pytest.raises(FileNotFoundError, match="ababab"):
            store_digest(payload, store)


def _write(directory: Path, payload: dict[str, Any]) -> Path:
    path = directory / "committed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
