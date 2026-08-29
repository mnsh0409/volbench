"""No tracked file may identify the person who wrote it.

IJF review is **double-blind**, and the reproducibility package is part of what
a reviewer sees. A home directory in a captured path, or a contact address in a
scraper's User-Agent, deanonymises the submission as surely as a name on the
title page — and both were in this repository until the sweep below was written.

**The patterns match shape, never a person.** That is not a stylistic
preference. A test that greps for a particular name or address must *contain*
that name or address, and it ships publicly, so it would leak precisely what it
exists to prevent. A shape catches the real problem and reveals nothing; if a
name-specific sweep is ever wanted it belongs in an uncommitted local script.

For the same reason the fixtures below are assembled from fragments at run
time. A literal home path in this file would make the sweep fail on its own
source, and excluding this file from its own check would put a blind spot
exactly where someone hiding something would put it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]

#: What identity looks like, by shape.
#:
#: * a POSIX home directory — the segment after ``/home/`` or ``/Users/`` is a
#:   login name, whoever it belongs to;
#: * the Windows equivalent;
#: * anything shaped like an email address.
#:
#: Deliberately not here: bare personal names, GitHub handles and institution
#: names. They cannot be matched by shape, only by listing them, and listing
#: them is the trap this file exists to avoid. They are a review question, not
#: a test.
PATTERNS: dict[str, re.Pattern[str]] = {
    "posix home directory": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "windows user directory": re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2}"),
}


def identity_leaks(paths: Iterable[Path]) -> list[str]:
    """``"<path>:<line>: <shape>: <matched text>"`` for every hit, readable."""
    found: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: nothing to read an identity out of
        for lineno, line in enumerate(text.splitlines(), 1):
            for shape, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match is not None:
                    found.append(f"{path}:{lineno}: {shape}: {match.group(0)}")
    return sorted(found)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


def tracked_files() -> list[Path]:
    return [REPO / rel for rel in _git("ls-files", "-z").stdout.split("\0") if rel]


needs_git = pytest.mark.skipif(
    shutil.which("git") is None or _git("rev-parse", "--git-dir").returncode != 0,
    reason="not a git checkout",
)

# Assembled, never written out — see the module docstring. Splitting on the
# literal that the pattern keys on is what keeps this file clean of its own
# matches, and the last test in the file proves that it is.
_HOME_SHAPE = "/" + "home" + "/somebody/volbench/.venv/bin/python"
_MAC_SHAPE = "/" + "Users" + "/somebody/volbench"
_WINDOWS_SHAPE = "D:" + "\\" + "Users" + "\\" + "somebody" + "\\" + "volbench"
_EMAIL_SHAPE = "somebody" + "@" + "example" + ".org"


class TestNoTrackedFileIdentifiesAnyone:
    @needs_git
    def test_the_repository_is_clean(self) -> None:
        leaks = identity_leaks(tracked_files())
        assert leaks == [], (
            "tracked files carry identifying paths or addresses, and this "
            "repository ships to a double-blind review:\n  " + "\n  ".join(leaks)
        )


class TestTheSweepCanFail:
    """The inert-proof. A sweep over a clean tree passes whether or not it
    works, so each shape is planted and the sweep is required to find it."""

    @pytest.mark.parametrize(
        ("shape", "planted"),
        [
            ("posix home directory", _HOME_SHAPE),
            ("posix home directory", _MAC_SHAPE),
            ("windows user directory", _WINDOWS_SHAPE),
            ("email address", _EMAIL_SHAPE),
        ],
    )
    def test_a_planted_identity_is_found(
        self, tmp_path: Path, shape: str, planted: str
    ) -> None:
        target = tmp_path / "captured_output.txt"
        target.write_text(f'  "executable": "{planted}"\n', encoding="utf-8")

        leaks = identity_leaks([target])
        assert len(leaks) == 1, leaks
        assert shape in leaks[0]

    def test_it_reports_where_it_found_it(self, tmp_path: Path) -> None:
        """A finding nobody can locate gets ignored, so the path and the line
        travel with it."""
        target = tmp_path / "notes.md"
        target.write_text(f"one\ntwo\nthe venv is at {_HOME_SHAPE}\n", encoding="utf-8")

        (leak,) = identity_leaks([target])
        assert str(target) in leak and ":3:" in leak

    def test_ordinary_text_is_not_flagged(self, tmp_path: Path) -> None:
        """A sweep that fires on anything gets switched off. Absolute paths
        that name no user, and version strings, must pass."""
        target = tmp_path / "fine.py"
        target.write_text(
            'ROOT = "/srv/volbench/cache"\n'
            'STORE = "data/grid_primary/store"\n'
            'UA = "volbench-research/0.1"\n'
            "python = '3.11.5'  # CPython\n",
            encoding="utf-8",
        )

        assert identity_leaks([target]) == []

    def test_binary_files_do_not_break_the_sweep(self, tmp_path: Path) -> None:
        target = tmp_path / "fragment.parquet"
        target.write_bytes(b"PAR1\x00\xff\xfe\x00garbage")

        assert identity_leaks([target]) == []

    def test_this_file_does_not_match_its_own_patterns(self) -> None:
        """The reason the fixtures above are assembled rather than written out.
        If this ever fails, do not exclude the file — restructure the literal."""
        assert identity_leaks([Path(__file__)]) == []
