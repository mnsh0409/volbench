"""The committed digest manifest, and the check that compares a tree against it.

``docs/P3_DATA_DIGESTS.json`` is the only place outside the gitignored store
where the study's input digests exist, so two things have to hold: the file has
to be well formed (checked here, since it is committed and CI can read it), and
the comparison has to *notice* — a checker that cannot fail is worse than no
checker, because it reports "verified" either way.

The panel itself is not rebuilt here: it needs ``data/``, which is not in a
clean checkout or in CI. Rebuilding is what ``--check`` does on a machine that
has the archives.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from volbench.benchmarks.data_digests import DEFAULT_MANIFEST, compare

SHA256 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(Path(DEFAULT_MANIFEST).read_text(encoding="utf-8"))


class TestTheCommittedManifest:
    def test_it_covers_the_whole_panel_once(self, manifest: dict[str, Any]) -> None:
        assets = [entry["asset"] for entry in manifest["assets"]]
        assert len(assets) == 11
        assert assets == sorted(set(assets))

    def test_every_digest_is_a_sha256_and_every_crypto_raw_is_null(
        self, manifest: dict[str, Any]
    ) -> None:
        """``raw_sha256`` is ``null`` exactly for the two exchange-API series:
        they are assembled from paged requests, not from one archived file."""
        for entry in manifest["assets"]:
            for key in ("series_sha256", "fit_series_sha256"):
                assert SHA256.match(entry[key]), f"{entry['asset']}.{key}"
            assert SHA256.match(entry["proxy"]["sha256"])
            assert (entry["raw_sha256"] is None) == (entry["role"] == "crypto")
            assert entry["n"] > 0

    def test_it_records_the_policy_the_fit_digest_depends_on(
        self, manifest: dict[str, Any]
    ) -> None:
        """A fit-series digest is only meaningful next to the D-018 policy that
        produced it, so the policy travels with it."""
        for entry in manifest["assets"]:
            assert entry["invalid_target_policy"] == "compact"

    def test_it_contains_no_data_only_digests_of_it(self, manifest: dict[str, Any]) -> None:
        """The licensing guard forbids tracking data under ``data/``; this file
        is under ``docs/`` and stays publishable because a digest is not the
        series. Nothing here may be a price, a return or a variance."""
        allowed = {
            "asset", "source", "role", "panel_start", "panel_end", "n", "raw_sha256",
            "series_sha256", "proxy", "fit_series_sha256", "invalid_target_policy",
        }
        for entry in manifest["assets"]:
            assert set(entry) == allowed
            assert set(entry["proxy"]) == {"name", "sha256"}


class TestCompareCanFail:
    def test_an_identical_pair_has_no_problems(self, manifest: dict[str, Any]) -> None:
        assert compare(manifest, copy.deepcopy(manifest)) == []

    @pytest.mark.parametrize(
        "key", ["series_sha256", "fit_series_sha256", "raw_sha256", "n"]
    )
    def test_a_moved_digest_is_reported_by_name(
        self, manifest: dict[str, Any], key: str
    ) -> None:
        rebuilt = copy.deepcopy(manifest)
        rebuilt["assets"][3][key] = "0" * 64 if key != "n" else -1
        problems = compare(manifest, rebuilt)
        assert len(problems) == 1
        assert manifest["assets"][3]["asset"] in problems[0] and key in problems[0]

    def test_a_moved_proxy_digest_is_reported(self, manifest: dict[str, Any]) -> None:
        rebuilt = copy.deepcopy(manifest)
        rebuilt["assets"][0]["proxy"]["sha256"] = "1" * 64
        assert len(compare(manifest, rebuilt)) == 1

    def test_a_panel_that_gained_or_lost_a_series_is_a_disagreement(
        self, manifest: dict[str, Any]
    ) -> None:
        rebuilt = copy.deepcopy(manifest)
        dropped = rebuilt["assets"].pop()
        assert any("absent" in line for line in compare(manifest, rebuilt))
        rebuilt["assets"].append({**dropped, "asset": "NEW"})
        problems = compare(manifest, rebuilt)
        assert any("absent" in line for line in problems)
        assert any("not in the manifest" in line for line in problems)
