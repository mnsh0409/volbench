"""Crisis sub-sample tags (D-004) — metadata about dates, never an input to fitting.

What a tag is
-------------
A label attached to a calendar date after the fact, used to *partition reported
scores* (docs/metrics_reference.md: "report per-regime alongside full-sample")
and to test H3 — that TSFM performance degrades in crisis sub-samples. It is
never a feature, never a regressor, never a filter on training data.

Why that separation is structural and not just a convention
-----------------------------------------------------------
Every window here is defined by dates that were only knowable *after* the
episode ended: labelling day ``t`` as "GFC" uses the fact that the crisis ran
to March 2009, which nobody knew in September 2008. A model that saw such a
tag at fit time would be reading the future — the exact failure CLAUDE.md rule 1
forbids. So this module:

- imports nothing from :mod:`volbench.models`, :mod:`volbench.evaluate` or
  :mod:`volbench.splitter` (pinned by ``tests/test_data_crisis.py``);
- returns only labels keyed by timestamp — it never touches prices, returns, or
  variance targets, and there is no function here that could hand a tag to a
  model;
- is applied to *result rows*, downstream of scoring, where the future is
  already past for every row involved.

Windows
-------
Verbatim from docs/research_design.md ("GFC Sep 08-Mar 09 · COVID Feb-Apr 20 ·
2022 tightening Jan-Oct 22 · Aug-2024 spike · latest 2025-26 stress window
(fixed at grid freeze)"), resolved to calendar-month boundaries. The fifth
window is **not** defined here: D-004 says it is fixed at grid freeze, so it
lives in :data:`PENDING_WINDOWS` with no dates and is excluded from tagging
until a human sets it. Inventing a range would be inventing a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

__all__ = [
    "CALM_TAG",
    "CRISIS_WINDOWS",
    "PENDING_WINDOWS",
    "CrisisWindow",
    "PendingWindow",
    "crisis_mask",
    "crisis_table",
    "tag_dates",
    "window_by_tag",
]

#: Label for a date inside no crisis window. Deliberately a real string rather
#: than NaN so a groupby over the tags never silently drops the calm sample.
CALM_TAG = "calm"


@dataclass(frozen=True)
class CrisisWindow:
    """One closed calendar window ``[start, end]``, both endpoints inclusive."""

    tag: str
    start: date
    end: date
    label: str
    #: What docs/research_design.md says, so the resolution to exact dates is
    #: auditable against its source instead of being folded away.
    source_phrase: str

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError("CrisisWindow.tag must be non-empty")
        if self.end < self.start:
            raise ValueError(f"{self.tag}: end {self.end} precedes start {self.start}")

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


@dataclass(frozen=True)
class PendingWindow:
    """A window D-004 names but deliberately leaves undated until grid freeze."""

    tag: str
    label: str
    source_phrase: str
    blocked_on: str


#: The four settled windows, in chronological order.
CRISIS_WINDOWS: tuple[CrisisWindow, ...] = (
    CrisisWindow(
        tag="gfc",
        start=date(2008, 9, 1),
        end=date(2009, 3, 31),
        label="Global financial crisis",
        source_phrase="GFC Sep 08-Mar 09",
    ),
    CrisisWindow(
        tag="covid",
        start=date(2020, 2, 1),
        end=date(2020, 4, 30),
        label="COVID-19 crash",
        source_phrase="COVID Feb-Apr 20",
    ),
    CrisisWindow(
        tag="tightening_2022",
        start=date(2022, 1, 1),
        end=date(2022, 10, 31),
        label="2022 monetary tightening",
        source_phrase="2022 tightening Jan-Oct 22",
    ),
    CrisisWindow(
        tag="spike_2024_08",
        start=date(2024, 8, 1),
        end=date(2024, 8, 31),
        label="August 2024 volatility spike",
        # research_design.md gives no day-level range for this one, only the
        # month. Read as the calendar month containing the 2024-08-05 unwind.
        # Flagged for confirmation in docs/PANEL_REPORT.md rather than presented
        # as settled.
        source_phrase="Aug-2024 spike",
    ),
)

#: Named by D-004, undated on purpose. Excluded from every tagging function
#: below; ``tests/test_data_crisis.py`` fails if a dated window is ever added
#: here or an undated one leaks into :data:`CRISIS_WINDOWS`.
PENDING_WINDOWS: tuple[PendingWindow, ...] = (
    PendingWindow(
        tag="stress_2025_26",
        label="Latest 2025-26 stress window",
        source_phrase="latest 2025-26 stress window (fixed at grid freeze)",
        blocked_on="D-004: dates are fixed at grid freeze, not before.",
    ),
)


def window_by_tag(tag: str) -> CrisisWindow:
    """Look up a settled window by tag, with a message that names the pending ones."""
    for window in CRISIS_WINDOWS:
        if window.tag == tag:
            return window
    pending = {w.tag for w in PENDING_WINDOWS}
    if tag in pending:
        blocked = next(w for w in PENDING_WINDOWS if w.tag == tag)
        raise KeyError(f"crisis window {tag!r} is not dated yet — {blocked.blocked_on}")
    raise KeyError(f"unknown crisis tag {tag!r}; known: {sorted(w.tag for w in CRISIS_WINDOWS)}")


def _dates(index: pd.DatetimeIndex) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("crisis tagging needs a pandas DatetimeIndex")
    if index.tz is None:
        raise ValueError("crisis tagging needs a tz-aware index (UTC)")
    return pd.Series(index.tz_convert("UTC").date, index=index)


def crisis_mask(index: pd.DatetimeIndex, tag: str) -> pd.Series:
    """Boolean Series flagging which timestamps in ``index`` fall inside ``tag``'s window."""
    window = window_by_tag(tag)
    days = _dates(index)
    mask: pd.Series = days.map(window.contains).astype(bool)
    return mask.rename(tag)


def tag_dates(index: pd.DatetimeIndex) -> pd.Series:
    """Map each timestamp to its crisis tag, or :data:`CALM_TAG` if in none.

    One tag per date: the settled windows do not overlap (asserted at import by
    ``tests/test_data_crisis.py``), so the first match is the only match.
    Returned as an ordered pandas ``Categorical`` so per-regime tables come out
    in chronological window order regardless of what the sample contains.
    """
    days = _dates(index)

    def lookup(day: date) -> str:
        for window in CRISIS_WINDOWS:
            if window.contains(day):
                return window.tag
        return CALM_TAG

    labels = days.map(lookup)
    categories = [CALM_TAG, *(w.tag for w in CRISIS_WINDOWS)]
    out: pd.Series = pd.Series(
        pd.Categorical(labels, categories=categories, ordered=True),
        index=index,
        name="regime",
    )
    return out


def crisis_table(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One boolean column per settled window, plus ``calm``, aligned to ``index``.

    Columns are mutually exclusive and jointly exhaustive — every row sums to
    exactly 1 — so summing a column counts that regime's observations without
    double-counting or losing any.
    """
    regime = tag_dates(index)
    columns = {CALM_TAG: regime == CALM_TAG}
    for window in CRISIS_WINDOWS:
        columns[window.tag] = regime == window.tag
    return pd.DataFrame(columns, index=index)
