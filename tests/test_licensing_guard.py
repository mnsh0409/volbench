"""Licensing guard: no raw or cached market data may ever enter the repository.

Stooq's terms forbid redistribution outright and Binance's are unresolved for
the derived series (docs/data_licenses.md), so volbench's position is that
*nothing* downloaded or hand-downloaded is committed — the adapters cache
locally under a gitignored tree and the package ships only synthetic fixtures.

These tests are the mechanical half of that promise. They ask git itself, not
a comment, whether the rule still holds.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]

#: Every path the data adapters write to. ``data/raw`` is where a human unpacks
#: the hand-downloaded Stooq archives; ``data/cache`` is where every adapter
#: writes parquet + SHA256 sidecars.
#: Trailing slashes are load-bearing: the .gitignore rule is ``/data/``, which
#: matches directories, and ``git check-ignore`` on a bare ``data`` cannot tell
#: that it is one when the (gitignored, hence often absent) directory does not
#: exist in this checkout.
DATA_PATHS = (
    "data/",
    "data/raw/",
    "data/cache/",
    "data/cache/stooq/",
    "data/cache/crypto/",
    "data/raw/stooq/d_us_txt/data/daily/us/nyse etfs/2/spy.us.txt",
    "data/cache/crypto/binance_btcusdt_1m_2024-01-02.parquet",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or _git("rev-parse", "--git-dir").returncode != 0,
    reason="not a git checkout",
)


class TestDataIsGitignored:
    @pytest.mark.parametrize("path", DATA_PATHS)
    def test_path_is_ignored(self, path: str) -> None:
        result = _git("check-ignore", "-q", "--no-index", path)
        assert result.returncode == 0, (
            f"{path!r} is NOT gitignored. Market data must never be committed "
            "(docs/data_licenses.md): Stooq forbids redistribution and Binance's "
            "terms are unresolved. Restore the /data/ rule in .gitignore."
        )

    def test_the_ignore_rule_is_root_anchored(self) -> None:
        # A bare "data/" would also swallow src/volbench/data/, i.e. the source
        # code of the data package. The rule must be anchored to the repo root.
        rules = [
            line.strip()
            for line in (REPO / ".gitignore").read_text().splitlines()
            if line.strip().rstrip("/").endswith("data") and not line.startswith("#")
        ]
        assert rules, "no data-directory rule in .gitignore at all"
        assert all(rule.startswith("/") for rule in rules), rules

    def test_source_package_is_not_ignored(self) -> None:
        # The mirror image of the rule above: the guard must not be so broad
        # that volbench's own data package stops being tracked.
        assert _git("check-ignore", "-q", "--no-index", "src/volbench/data").returncode != 0


class TestNoDataIsTracked:
    def test_nothing_under_data_is_in_the_index(self) -> None:
        tracked = _git("ls-files", "--", "data/").stdout.split()
        assert tracked == [], f"market data files are staged/committed: {tracked[:10]}"

    def test_no_archive_or_parquet_data_files_anywhere(self) -> None:
        # Catches a copy smuggled in outside data/ — e.g. a "sample" parquet or
        # a raw Stooq .txt dropped next to the code.
        tracked = _git("ls-files").stdout.split()
        offenders = [
            path
            for path in tracked
            if path.endswith((".parquet", ".zip"))
            or (path.endswith(".txt") and "/daily/" in path)
        ]
        assert offenders == [], f"data-shaped files are tracked: {offenders}"

    def test_committed_fixtures_stay_tiny(self) -> None:
        # A real archive slice would be large; the fixtures are hand-written
        # synthetic bars and must stay that way.
        for fixture in (REPO / "tests" / "fixtures").iterdir():
            size = fixture.stat().st_size
            assert size < 64_000, f"{fixture.name} is {size} bytes — is that real data?"


class TestAdaptersDefaultInsideTheIgnoredTree:
    def test_panel_defaults_point_at_the_gitignored_paths(self) -> None:
        from volbench.data.panel import DEFAULT_CACHE_ROOT, DEFAULT_RAW_ROOT

        for default in (DEFAULT_RAW_ROOT, DEFAULT_CACHE_ROOT):
            assert _git("check-ignore", "-q", "--no-index", f"{default}/").returncode == 0, (
                f"{default} is a default write target but is not gitignored"
            )
