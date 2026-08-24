"""Crisis sub-sample tags: correct windows, and structurally unable to reach a model.

The second half matters as much as the first. A crisis label is defined by
dates that were only knowable after the episode ended, so a tag reaching a
model at fit time would be look-ahead of the plainest kind (CLAUDE.md rule 1).
These tests pin that the module cannot do so, rather than trusting the comment
that says it doesn't.
"""

from __future__ import annotations

import ast
from datetime import date
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from volbench.data.crisis import (
    CALM_TAG,
    CRISIS_WINDOWS,
    PENDING_WINDOWS,
    CrisisWindow,
    crisis_mask,
    crisis_table,
    tag_dates,
    window_by_tag,
)

CRISIS_SOURCE = Path(__file__).parents[1] / "src" / "volbench" / "data" / "crisis.py"


def _index(*days: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in days])


class TestWindowDefinitions:
    def test_the_four_settled_windows_are_present(self) -> None:
        assert [w.tag for w in CRISIS_WINDOWS] == [
            "gfc", "covid", "tightening_2022", "spike_2024_08",
        ]

    def test_windows_match_research_design(self) -> None:
        # docs/research_design.md: "GFC Sep 08-Mar 09 · COVID Feb-Apr 20 ·
        # 2022 tightening Jan-Oct 22 · Aug-2024 spike".
        expected = {
            "gfc": (date(2008, 9, 1), date(2009, 3, 31)),
            "covid": (date(2020, 2, 1), date(2020, 4, 30)),
            "tightening_2022": (date(2022, 1, 1), date(2022, 10, 31)),
            "spike_2024_08": (date(2024, 8, 1), date(2024, 8, 31)),
        }
        assert {w.tag: (w.start, w.end) for w in CRISIS_WINDOWS} == expected

    def test_windows_are_chronological_and_disjoint(self) -> None:
        # tag_dates returns the first match; overlapping windows would make
        # the label depend on declaration order rather than on the date.
        for earlier, later in pairwise(CRISIS_WINDOWS):
            assert earlier.end < later.start

    def test_each_window_cites_its_source_phrase(self) -> None:
        for window in CRISIS_WINDOWS:
            assert window.source_phrase, f"{window.tag} has no source phrase"

    def test_the_2025_26_window_is_pending_not_invented(self) -> None:
        # D-004 fixes it at grid freeze. Guessing a range would fabricate a
        # sub-sample result.
        assert [w.tag for w in PENDING_WINDOWS] == ["stress_2025_26"]
        settled = {w.tag for w in CRISIS_WINDOWS}
        assert not settled & {w.tag for w in PENDING_WINDOWS}

    def test_pending_window_lookup_explains_itself(self) -> None:
        with pytest.raises(KeyError, match="not dated yet"):
            window_by_tag("stress_2025_26")

    def test_unknown_tag_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="unknown crisis tag"):
            window_by_tag("nope")

    def test_backwards_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="precedes start"):
            CrisisWindow(
                tag="x", start=date(2020, 5, 1), end=date(2020, 1, 1),
                label="x", source_phrase="x",
            )

    def test_empty_tag_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CrisisWindow(
                tag="", start=date(2020, 1, 1), end=date(2020, 5, 1),
                label="x", source_phrase="x",
            )


class TestTagging:
    def test_endpoints_are_inclusive(self) -> None:
        tags = tag_dates(_index("2008-09-01", "2009-03-31"))
        assert list(tags) == ["gfc", "gfc"]

    def test_days_just_outside_are_calm(self) -> None:
        tags = tag_dates(_index("2008-08-31", "2009-04-01"))
        assert list(tags) == [CALM_TAG, CALM_TAG]

    def test_each_window_is_reachable(self) -> None:
        tags = tag_dates(_index("2008-10-15", "2020-03-16", "2022-06-01", "2024-08-05"))
        assert list(tags) == ["gfc", "covid", "tightening_2022", "spike_2024_08"]

    def test_regime_categories_are_chronological(self) -> None:
        tags = tag_dates(_index("2024-08-05", "2008-10-15"))
        assert list(tags.cat.categories) == [
            CALM_TAG, "gfc", "covid", "tightening_2022", "spike_2024_08",
        ]

    def test_tagging_is_index_aligned(self) -> None:
        index = _index("2019-01-02", "2020-03-16", "2021-01-04")
        tags = tag_dates(index)
        pd.testing.assert_index_equal(tags.index, index)

    def test_tagging_depends_only_on_the_date_not_on_order(self) -> None:
        index = _index("2020-03-16", "2008-10-15", "2015-06-01")
        assert list(tag_dates(index)) == ["covid", "gfc", CALM_TAG]

    def test_naive_index_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            tag_dates(pd.DatetimeIndex([pd.Timestamp("2020-03-16")]))

    def test_non_datetime_index_is_refused(self) -> None:
        with pytest.raises(TypeError, match="DatetimeIndex"):
            tag_dates(pd.Index([1, 2, 3]))  # type: ignore[arg-type]

    def test_tagging_respects_the_calendar_date_in_utc(self) -> None:
        # A timestamp late on the last GFC day is still in the window.
        tags = tag_dates(pd.DatetimeIndex([pd.Timestamp("2009-03-31 23:59", tz="UTC")]))
        assert list(tags) == ["gfc"]


class TestMasksAndTable:
    def test_mask_matches_the_window(self) -> None:
        mask = crisis_mask(_index("2020-01-31", "2020-02-03", "2020-05-01"), "covid")
        assert list(mask) == [False, True, False]

    def test_table_columns_are_exhaustive_and_exclusive(self) -> None:
        index = _index("2008-10-15", "2015-06-01", "2020-03-16", "2022-06-01", "2024-08-05")
        table = crisis_table(index)
        assert list(table.columns) == [
            CALM_TAG, "gfc", "covid", "tightening_2022", "spike_2024_08",
        ]
        assert (table.sum(axis=1) == 1).all(), "every day belongs to exactly one regime"

    def test_table_counts_match_tag_counts(self) -> None:
        index = pd.date_range("2008-01-01", "2010-01-01", freq="D", tz="UTC")
        table = crisis_table(index)
        tags = tag_dates(index)
        for column in table.columns:
            assert int(table[column].sum()) == int((tags == column).sum())


class TestTagsCannotReachAModel:
    """Structural guarantees, checked against the module's own AST."""

    def _imports(self) -> set[str]:
        tree = ast.parse(CRISIS_SOURCE.read_text())
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        return modules

    def test_does_not_import_the_forecasting_stack(self) -> None:
        forbidden = {
            "volbench.models",
            "volbench.evaluate",
            "volbench.splitter",
            "volbench.dist",
        }
        assert not (self._imports() & forbidden)

    def test_does_not_import_any_data_adapter(self) -> None:
        # Tags are about dates. If this module could read prices, a future
        # edit could hand a tag to something that fits on them.
        assert not any(m.startswith("volbench.data.") for m in self._imports())

    def test_public_api_takes_only_an_index(self) -> None:
        # Nothing here accepts prices, returns, or a variance target, so
        # there is no signature through which a tag could enter a fit.
        tree = ast.parse(CRISIS_SOURCE.read_text())
        public = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        }
        assert set(public) == {"window_by_tag", "crisis_mask", "tag_dates", "crisis_table"}
        for name, node in public.items():
            args = [a.arg for a in node.args.args]
            assert args in (["index"], ["index", "tag"], ["tag"]), f"{name}{args}"
