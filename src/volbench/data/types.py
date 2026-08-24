"""Canonical single-asset price container.

Every data adapter (stooq, crypto, byo, ...) emits :class:`TimeSeriesFrame`
objects. Trading calendars differ per asset (US equities vs. European
indices vs. 24/7 crypto), so this module never reindexes or joins two assets
onto a shared calendar — that decision belongs to whatever consumes multiple
frames downstream, not to the data layer (docs/design.md §Components).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

__all__ = ["CLOSE_COLUMN", "OHLC_COLUMNS", "TimeSeriesFrame"]

OHLC_COLUMNS = ("open", "high", "low", "close")
CLOSE_COLUMN = ("close",)


@dataclass(frozen=True)
class TimeSeriesFrame:
    """An immutable OHLC(-or-close) price series on its asset's native calendar.

    Parameters
    ----------
    data:
        A :class:`pandas.DataFrame` with a tz-aware :class:`~pandas.DatetimeIndex`
        and either a ``close`` column or the full ``open``/``high``/``low``/``close``
        set (extra columns such as ``volume`` are carried through unvalidated).
    asset_id:
        Stable identifier for the asset (e.g. ``"SPX"``, ``"BTC-USD"``).
    source:
        Tag naming the adapter/provenance (e.g. ``"stooq"``, ``"binance"``, ``"byo"``).

    Validation (raised as :class:`ValueError`/:class:`TypeError` at construction):
    the index must be tz-aware, non-empty, strictly increasing, free of duplicate
    timestamps, and the required price columns must contain no NaN. The index is
    normalized to UTC; the data is defensively copied so external mutation of the
    caller's DataFrame cannot alter this frame after construction (the reverse is
    not enforced — treat ``.data`` itself as read-only by convention).
    """

    data: pd.DataFrame
    asset_id: str
    source: str

    def __post_init__(self) -> None:
        index = self.data.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("TimeSeriesFrame.data must have a pandas DatetimeIndex")
        if index.tz is None:
            raise ValueError("TimeSeriesFrame index must be tz-aware (UTC)")
        if index.size == 0:
            raise ValueError("TimeSeriesFrame requires at least one observation")
        if index.has_duplicates:
            raise ValueError("TimeSeriesFrame index has duplicate timestamps")
        if not index.is_monotonic_increasing:
            raise ValueError("TimeSeriesFrame index must be strictly increasing")

        columns = set(self.data.columns)
        required: tuple[str, ...]
        if set(OHLC_COLUMNS).issubset(columns):
            required = OHLC_COLUMNS
        elif "close" in columns:
            required = CLOSE_COLUMN
        else:
            raise ValueError(
                "TimeSeriesFrame requires a 'close' column, or the full "
                f"open/high/low/close set; got columns={sorted(columns)}"
            )
        if self.data[list(required)].isna().to_numpy().any():
            raise ValueError(f"TimeSeriesFrame required columns {required} contain NaN")

        if not self.asset_id:
            raise ValueError("TimeSeriesFrame.asset_id must be non-empty")
        if not self.source:
            raise ValueError("TimeSeriesFrame.source must be non-empty")

        frame = self.data.copy(deep=True)
        frame.index = index.tz_convert("UTC")  # `index` was validated above as a DatetimeIndex
        frame.index.name = "timestamp"
        object.__setattr__(self, "data", frame)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def index(self) -> pd.DatetimeIndex:
        # Enforced in __post_init__; the stubs only know that an index is an Index.
        return cast(pd.DatetimeIndex, self.data.index)

    @property
    def has_ohlc(self) -> bool:
        return set(OHLC_COLUMNS).issubset(self.data.columns)

    def _column(self, name: str) -> pd.Series:
        if name not in self.data.columns:
            raise ValueError(f"TimeSeriesFrame for {self.asset_id!r} has no {name!r} column")
        series: pd.Series = self.data[name]
        return series

    @property
    def close(self) -> pd.Series:
        return self._column("close")

    @property
    def open(self) -> pd.Series:
        return self._column("open")

    @property
    def high(self) -> pd.Series:
        return self._column("high")

    @property
    def low(self) -> pd.Series:
        return self._column("low")
