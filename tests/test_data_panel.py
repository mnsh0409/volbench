"""Panel builder: specs, file resolution, bar repair, target construction, trimming.

No test here touches the real archives under ``data/raw`` — that tree is
gitignored and absent from CI. Every fixture is a synthetic file written into
``tmp_path``, which also exercises the configurable ``raw_root`` that exists
precisely so the panel is not welded to one machine's layout.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volbench.data.crisis import CRISIS_WINDOWS
from volbench.data.diagnostics import crisis_coverage
from volbench.data.panel import (
    CRYPTO_PANEL,
    DEFAULT_REPAIR_TOLERANCE,
    EQUITY_PANEL,
    PANEL_END,
    PANEL_START,
    TARGET_NAMES,
    BarQuality,
    EquitySpec,
    PanelSeries,
    build_equity_series,
    build_targets,
    daily_bars_from_minutes,
    repair_bars,
    resolve_equity_path,
)
from volbench.data.proxies import overnight_plus_range_variance
from volbench.data.types import TimeSeriesFrame
from volbench.splitter import RollingOriginSplitter

BULK_FIXTURE = Path(__file__).parent / "fixtures" / "stooq_bulk_sample.txt"

Row = tuple[str, float, float, float, float]


def _bulk_text(ticker: str, rows: list[Row]) -> str:
    header = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"
    lines = [header]
    for day, o, h, low, c in rows:
        lines.append(f"{ticker},D,{day},000000,{o},{h},{low},{c},1000,0")
    return "\n".join(lines) + "\n"


def _synthetic_rows(n: int, start: str = "2004-11-01") -> list[Row]:
    """A deterministic well-formed OHLC path on business days."""
    rng = np.random.default_rng(20260825)
    days = pd.bdate_range(start, periods=n, tz="UTC")
    rows = []
    price = 100.0
    for day in days:
        o = price * float(np.exp(rng.normal(0, 0.004)))
        c = o * float(np.exp(rng.normal(0, 0.008)))
        h = max(o, c) * float(np.exp(abs(rng.normal(0, 0.004))))
        low = min(o, c) * float(np.exp(-abs(rng.normal(0, 0.004))))
        rows.append((day.strftime("%Y%m%d"), round(o, 4), round(h, 4), round(low, 4), round(c, 4)))
        price = c
    return rows


class TestPanelSpecs:
    def test_ten_equity_series_per_d012(self) -> None:
        assert len(EQUITY_PANEL) == 10
        assert set(EQUITY_PANEL) == {
            "NDX", "DAX", "CAC", "NKX", "HSI", "TWSE", "KOSPI", "SPY", "DIA", "ISF",
        }

    def test_seven_direct_indices_and_three_etf_proxies(self) -> None:
        roles = [spec.role for spec in EQUITY_PANEL.values()]
        assert roles.count("index") == 7
        assert roles.count("etf_proxy") == 3

    def test_every_etf_proxy_records_what_it_stands_in_for(self) -> None:
        # D-012 substituted tradable ETFs for the SPX/DJI/FTSE slots Stooq
        # retired. A proxy that forgot its referent would silently become "an
        # index" in every table downstream.
        for spec in EQUITY_PANEL.values():
            if spec.role == "etf_proxy":
                assert spec.proxy_for, f"{spec.asset_id} is a proxy for nothing"
            else:
                assert spec.proxy_for is None

    def test_asset_ids_are_their_own_keys(self) -> None:
        for asset_id, spec in EQUITY_PANEL.items():
            assert spec.asset_id == asset_id
        for asset_id, spec in CRYPTO_PANEL.items():
            assert spec.asset_id == asset_id

    def test_panel_window_matches_d004(self) -> None:
        assert pd.Timestamp("2005-01-01", tz="UTC") == PANEL_START
        assert PANEL_END > PANEL_START

    def test_module_never_reaches_stooq_over_the_network(self) -> None:
        """stooq.com is hand-download-only (docs/data_licenses.md).

        Structural, not aspirational: the panel module must not import any
        network entry point, so no future edit can quietly add a fetch.
        """
        source = (Path(__file__).parents[1] / "src" / "volbench" / "data" / "panel.py").read_text()
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
        assert "requests" not in imported
        for banned in ("fetch_stooq_csv", "download_index"):
            assert banned not in imported, f"panel.py must not import {banned}"
        # The prose *documents* the rule, so look for an actual URL, not the word.
        assert "//stooq" not in source


class TestResolveEquityPath:
    def test_finds_file_at_the_recorded_layout(self, tmp_path: Path) -> None:
        spec = EQUITY_PANEL["NDX"]
        target = tmp_path / spec.relative_path
        target.parent.mkdir(parents=True)
        target.write_text("x")
        assert resolve_equity_path(spec, tmp_path) == target

    def test_falls_back_to_a_filename_search(self, tmp_path: Path) -> None:
        # A bulk zip re-extracted with a different tool can add or drop a
        # directory level; that must not kill the panel.
        spec = EQUITY_PANEL["NDX"]
        moved = tmp_path / "somewhere" / "else" / Path(spec.relative_path).name
        moved.parent.mkdir(parents=True)
        moved.write_text("x")
        assert resolve_equity_path(spec, tmp_path) == moved

    def test_missing_file_names_the_manual_download_rule(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="never downloaded programmatically"):
            resolve_equity_path(EQUITY_PANEL["NDX"], tmp_path)

    def test_ambiguous_match_raises_instead_of_guessing(self, tmp_path: Path) -> None:
        spec = EQUITY_PANEL["NDX"]
        name = Path(spec.relative_path).name
        for sub in ("a", "b"):
            (tmp_path / sub).mkdir()
            (tmp_path / sub / name).write_text("x")
        with pytest.raises(FileNotFoundError, match="ambiguous"):
            resolve_equity_path(spec, tmp_path)


class TestRepairBars:
    def _frame(self, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
        index = pd.date_range("2020-01-01", periods=len(rows), freq="D", tz="UTC")
        return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)

    def test_clean_bars_are_untouched(self) -> None:
        frame = self._frame([(10.0, 11.0, 9.0, 10.5), (10.5, 10.9, 10.1, 10.2)])
        out, flag, quality = repair_bars(frame)
        pd.testing.assert_frame_equal(out, frame)
        assert not flag.any()
        assert (quality.repaired, quality.inconsistent) == (0, 0)

    def test_rounding_scale_violation_is_clamped_and_counted(self) -> None:
        # close 1e-6 above the high: the source file's own decimal rounding.
        frame = self._frame([(10.0, 10.0, 9.0, 10.00001)])
        out, flag, quality = repair_bars(frame, tolerance=1e-5)
        assert quality.repaired == 1
        assert quality.inconsistent == 0
        assert not flag.iloc[0]
        assert out["high"].iloc[0] == pytest.approx(10.00001)
        assert out["low"].iloc[0] == pytest.approx(9.0)

    def test_material_violation_is_flagged_not_rewritten(self) -> None:
        # close 1% above the high: a genuine feed disagreement. The bar must
        # survive unmodified and be flagged, never silently "fixed".
        frame = self._frame([(10.0, 10.0, 9.0, 10.1)])
        out, flag, quality = repair_bars(frame, tolerance=1e-5)
        assert quality.inconsistent == 1
        assert quality.repaired == 0
        assert bool(flag.iloc[0])
        assert out["high"].iloc[0] == pytest.approx(10.0), "a real error must not be clamped away"

    def test_zero_range_days_are_counted(self) -> None:
        frame = self._frame([(10.0, 10.0, 10.0, 10.0), (10.0, 11.0, 9.0, 10.0)])
        _, _, quality = repair_bars(frame)
        assert quality.zero_range == 1

    def test_requires_full_ohlc(self) -> None:
        index = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
        with pytest.raises(ValueError, match="missing"):
            repair_bars(pd.DataFrame({"close": [1.0, 2.0]}, index=index))

    def test_negative_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            repair_bars(self._frame([(10.0, 11.0, 9.0, 10.5)]), tolerance=-1.0)


class TestBuildTargets:
    def _frame(self, n: int = 40) -> pd.DataFrame:
        rows = _synthetic_rows(n)
        index = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d, *_ in rows], name="timestamp")
        return pd.DataFrame(
            [(o, h, low, c) for _, o, h, low, c in rows],
            columns=["open", "high", "low", "close"],
            index=index,
        )

    def test_produces_the_four_targets_in_order(self) -> None:
        targets, components = build_targets(self._frame())
        assert tuple(targets.columns) == TARGET_NAMES
        assert list(components.columns) == ["overnight_variance", "rogers_satchell"]

    def test_primary_target_equals_the_d016_estimator(self) -> None:
        frame = self._frame()
        targets, _ = build_targets(frame)
        expected = overnight_plus_range_variance(
            frame["open"], frame["high"], frame["low"], frame["close"]
        )
        pd.testing.assert_series_equal(
            targets["overnight_plus_range"], expected, check_names=False
        )

    def test_components_sum_to_the_primary_target(self) -> None:
        targets, components = build_targets(self._frame())
        total = components["overnight_variance"] + components["rogers_satchell"]
        pd.testing.assert_series_equal(
            total, targets["overnight_plus_range"], check_names=False
        )

    def test_first_observation_has_no_overnight_term(self) -> None:
        targets, _ = build_targets(self._frame())
        assert np.isnan(targets["overnight_plus_range"].iloc[0])
        assert np.isnan(targets["squared_return"].iloc[0])
        assert not np.isnan(targets["parkinson"].iloc[0]), "parkinson needs no previous close"

    def test_targets_are_non_negative(self) -> None:
        targets, _ = build_targets(self._frame(200))
        for name in ("overnight_plus_range", "parkinson", "garman_klass", "squared_return"):
            values = targets[name].dropna()
            assert (values >= 0).all(), name

    def test_inconsistent_bars_nan_the_range_targets_but_not_squared_return(self) -> None:
        frame = self._frame()
        flag = pd.Series(False, index=frame.index)
        flag.iloc[5] = True
        targets, components = build_targets(frame, inconsistent=flag)
        for name in ("overnight_plus_range", "parkinson", "garman_klass"):
            assert np.isnan(targets[name].iloc[5]), name
        assert not np.isnan(targets["squared_return"].iloc[5])
        assert np.isnan(components["rogers_satchell"].iloc[5])

    def test_rows_are_never_dropped(self) -> None:
        frame = self._frame()
        flag = pd.Series(True, index=frame.index)
        targets, _ = build_targets(frame, inconsistent=flag)
        assert len(targets) == len(frame)
        pd.testing.assert_index_equal(targets.index, frame.index)


class TestBuildEquitySeries:
    def _write(self, root: Path, spec: EquitySpec, rows: list) -> Path:
        path = root / spec.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_bulk_text(spec.ticker, rows))
        return path

    def test_builds_targets_and_trims_to_the_panel_window(self, tmp_path: Path) -> None:
        spec = EQUITY_PANEL["NDX"]
        # Starts in 2004, i.e. before the panel window opens.
        self._write(tmp_path / "raw", spec, _synthetic_rows(120, start="2004-11-01"))
        series = build_equity_series(
            "NDX", raw_root=tmp_path / "raw", cache_root=tmp_path / "cache"
        )
        assert series.index[0] >= PANEL_START
        assert series.archive_start < PANEL_START
        pd.testing.assert_index_equal(series.targets.index, series.index)
        pd.testing.assert_index_equal(series.components.index, series.index)

    def test_pre_window_history_supplies_the_first_overnight_term(self, tmp_path: Path) -> None:
        """The first in-window day must keep a real previous close.

        Trimming before building the target would NaN it instead — a silent
        loss of a scorable day. Reading the previous close is strictly
        backward-looking, so this costs nothing in temporal integrity.
        """
        spec = EQUITY_PANEL["NDX"]
        self._write(tmp_path / "raw", spec, _synthetic_rows(120, start="2004-11-01"))
        series = build_equity_series(
            "NDX", raw_root=tmp_path / "raw", cache_root=tmp_path / "cache"
        )
        assert not np.isnan(series.primary.iloc[0])

    def test_a_series_starting_inside_the_window_has_a_nan_first_day(self, tmp_path: Path) -> None:
        spec = EQUITY_PANEL["SPY"]
        self._write(tmp_path / "raw", spec, _synthetic_rows(60, start="2005-03-01"))
        series = build_equity_series(
            "SPY", raw_root=tmp_path / "raw", cache_root=tmp_path / "cache"
        )
        assert np.isnan(series.primary.iloc[0]), "no previous close exists before the file starts"

    def test_wrong_ticker_in_the_file_is_refused(self, tmp_path: Path) -> None:
        # Guards against grabbing the wrong file out of an archive holding
        # tens of thousands of symbols.
        spec = EQUITY_PANEL["NDX"]
        path = (tmp_path / "raw") / spec.relative_path
        path.parent.mkdir(parents=True)
        path.write_text(_bulk_text("^WRONG", _synthetic_rows(30)))
        with pytest.raises(Exception, match="declares ticker"):
            build_equity_series("NDX", raw_root=tmp_path / "raw", cache_root=tmp_path / "cache")

    def test_unknown_asset_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="unknown equity asset_id"):
            build_equity_series("NOPE", raw_root=tmp_path, cache_root=tmp_path)

    def test_empty_window_raises_with_the_file_span(self, tmp_path: Path) -> None:
        spec = EQUITY_PANEL["NDX"]
        self._write(tmp_path / "raw", spec, _synthetic_rows(30, start="1990-01-01"))
        with pytest.raises(ValueError, match="no observations in"):
            build_equity_series(
                "NDX", raw_root=tmp_path / "raw", cache_root=tmp_path / "cache"
            )

    def test_etf_proxy_carries_the_d012_substitution_note(self, tmp_path: Path) -> None:
        spec = EQUITY_PANEL["SPY"]
        self._write(tmp_path / "raw", spec, _synthetic_rows(60, start="2005-03-01"))
        series = build_equity_series(
            "SPY", raw_root=tmp_path / "raw", cache_root=tmp_path / "cache"
        )
        assert any("D-012" in note for note in series.notes)
        assert series.role == "etf_proxy"


class TestDailyBarsFromMinutes:
    def _minutes(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-01 23:55", periods=12, freq="1min", tz="UTC")
        return pd.DataFrame(
            {
                "open": np.arange(12, dtype=float) + 100.0,
                "high": np.arange(12, dtype=float) + 100.5,
                "low": np.arange(12, dtype=float) + 99.5,
                "close": np.arange(12, dtype=float) + 100.2,
                "volume": np.ones(12),
            },
            index=index,
        )

    def test_day_boundaries_never_straddle_midnight(self) -> None:
        daily = daily_bars_from_minutes(self._minutes())
        assert len(daily) == 2
        first, second = daily.index[0], daily.index[1]
        assert first == pd.Timestamp("2024-01-01", tz="UTC")
        # Day 1 holds only its own five bars (23:55..23:59); its close must be
        # the last bar *within* the day, never the first bar of the next.
        assert daily.loc[first, "close"] == pytest.approx(104.2)
        assert daily.loc[second, "open"] == pytest.approx(105.0)

    def test_ohlc_are_first_max_min_last(self) -> None:
        daily = daily_bars_from_minutes(self._minutes())
        day2 = daily.index[1]
        assert daily.loc[day2, "high"] == pytest.approx(111.5)
        assert daily.loc[day2, "low"] == pytest.approx(104.5)
        assert daily.loc[day2, "volume"] == pytest.approx(7.0)

    def test_rejects_naive_index(self) -> None:
        bars = self._minutes()
        bars.index = bars.index.tz_localize(None)
        with pytest.raises(ValueError, match="tz-aware"):
            daily_bars_from_minutes(bars)

    def test_rejects_unsorted_index(self) -> None:
        bars = self._minutes()
        with pytest.raises(ValueError, match="strictly increasing"):
            daily_bars_from_minutes(bars.iloc[::-1])


class TestBulkFixture:
    def test_default_tolerance_is_small(self) -> None:
        assert 0 < DEFAULT_REPAIR_TOLERANCE <= 1e-4

    def test_committed_fixture_is_synthetic(self) -> None:
        # Stooq data may not be redistributed (docs/data_licenses.md), so the
        # committed fixture must not be real market data.
        assert "FAKE.US" in BULK_FIXTURE.read_text()


class TestRepairDoesNotCorruptItsOwnDiagnostics:
    """Regression: the counts must describe the source bar, not the repaired one.

    ``DataFrame.to_numpy()`` on a single-dtype frame may return a view of the
    frame's buffer. When ``repair_bars`` clamped a bar through ``frame.loc``,
    the classification arrays changed underneath it, and a ``high == low`` day
    that a sub-tolerance repair then widened stopped being counted as
    zero-range. Found on the real NKX archive (2020-10-01, a bar with
    ``O=H=L=23184.93`` and ``C=23185.12``), where the reported zero-range count
    was 0 instead of 1.
    """

    def test_a_repaired_flat_bar_is_still_counted_as_zero_range(self) -> None:
        index = pd.date_range("2020-10-01", periods=1, freq="D", tz="UTC")
        # Exactly the NKX bar: flat OHL, close a hair above, within tolerance.
        frame = pd.DataFrame(
            [(23184.93, 23184.93, 23184.93, 23185.12)],
            columns=["open", "high", "low", "close"],
            index=index,
        )
        out, _, quality = repair_bars(frame, tolerance=DEFAULT_REPAIR_TOLERANCE)
        assert quality.repaired == 1, "the 8e-6 violation is within tolerance"
        assert quality.zero_range == 1, "the source bar had high == low"
        assert out["high"].iloc[0] == pytest.approx(23185.12), "and it was clamped"

    def test_the_input_frame_is_never_mutated(self) -> None:
        index = pd.date_range("2020-10-01", periods=1, freq="D", tz="UTC")
        frame = pd.DataFrame(
            [(10.0, 10.0, 10.0, 10.00001)],
            columns=["open", "high", "low", "close"],
            index=index,
        )
        before = frame.copy()
        repair_bars(frame)
        pd.testing.assert_frame_equal(frame, before)


class TestFutureDataCannotReachAnEarlierTarget:
    """The leakage-check canary, applied to target construction.

    Corrupt every bar strictly after date T and rebuild: every target dated
    <= T must be **bit-identical**. The panel builder is where a calendar
    mistake would first admit the future — it reads a previous close across the
    panel boundary, aggregates 24/7 bars into days, and masks bad bars — so the
    canary belongs here as well as in the end-to-end smoke test.
    """

    def _build(self, tmp_path: Path, rows: list[Row], tag: str) -> pd.DataFrame:
        spec = EQUITY_PANEL["NDX"]
        root = tmp_path / tag / "raw"
        path = root / spec.relative_path
        path.parent.mkdir(parents=True)
        path.write_text(_bulk_text(spec.ticker, rows))
        series = build_equity_series(
            "NDX", raw_root=root, cache_root=tmp_path / tag / "cache"
        )
        return series.targets

    def test_corrupting_the_future_leaves_earlier_targets_bit_identical(
        self, tmp_path: Path
    ) -> None:
        rows = _synthetic_rows(400, start="2004-11-01")
        cutoff = 250

        corrupted = list(rows)
        for i in range(cutoff + 1, len(rows)):
            day, o, h, low, c = rows[i]
            # A gross, unmistakable corruption: 3x every price after the cutoff.
            corrupted[i] = (day, o * 3, h * 3, low * 3, c * 3)

        clean_targets = self._build(tmp_path, rows, "clean")
        dirty_targets = self._build(tmp_path, corrupted, "dirty")

        cutoff_day = pd.Timestamp(rows[cutoff][0], tz="UTC")
        clean_past = clean_targets.loc[clean_targets.index <= cutoff_day]
        dirty_past = dirty_targets.loc[dirty_targets.index <= cutoff_day]

        assert not clean_past.empty
        pd.testing.assert_frame_equal(clean_past, dirty_past)

    def test_the_canary_can_actually_fail(self, tmp_path: Path) -> None:
        # A canary that cannot fire proves nothing: corrupting from an EARLIER
        # date must change the same rows the test above asserts are stable.
        rows = _synthetic_rows(400, start="2004-11-01")
        corrupted = list(rows)
        for i in range(100, len(rows)):
            day, o, h, low, c = rows[i]
            corrupted[i] = (day, o * 3, h * 3, low * 3, c * 3)

        cutoff_day = pd.Timestamp(rows[250][0], tz="UTC")
        clean = self._build(tmp_path, rows, "clean2")
        dirty = self._build(tmp_path, corrupted, "dirty2")
        clean_past = clean.loc[clean.index <= cutoff_day]
        dirty_past = dirty.loc[dirty.index <= cutoff_day]
        assert not clean_past.equals(dirty_past)


class TestCrisisCoverageUsesTheSplitter:
    """Coverage counts must come from the splitter, not from re-derived arithmetic."""

    def _panel(self, n: int) -> dict[str, PanelSeries]:
        rows = _synthetic_rows(n, start="2004-11-01")
        index = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d, *_ in rows])
        data = pd.DataFrame(
            [(o, h, low, c) for _, o, h, low, c in rows],
            columns=["open", "high", "low", "close"],
            index=index,
        )
        targets, components = build_targets(data)
        return {
            "X": PanelSeries(
                asset_id="X",
                source="test",
                role="index",
                description="synthetic",
                frame=TimeSeriesFrame(data=data, asset_id="X", source="test"),
                targets=targets,
                components=components,
                quality=BarQuality(len(data), 0, 0, 0, 0),
                primary_target="overnight_plus_range",
                archive_start=index[0],
                archive_end=index[-1],
            )
        }

    def test_first_scored_position_matches_the_splitter(self) -> None:
        # Regression: the first scored index is `window`, because the splitter's
        # first origin is `window - 1` and its test set is `origin + 1`. An
        # earlier version used `window + horizon` and lost one observation.
        window, n = 50, 200
        panel = self._panel(n)
        coverage = crisis_coverage(panel, window=window)
        first_test = next(RollingOriginSplitter(window=window).split(n)).test[0]
        assert coverage.loc["X", "scored_from"] == panel["X"].index[first_test].date()
        assert int(coverage.loc["X", "n_scored"]) == n - window

    def test_a_series_too_short_to_split_scores_nothing(self) -> None:
        coverage = crisis_coverage(self._panel(30), window=50)
        assert int(coverage.loc["X", "n_scored"]) == 0
        # pandas coerces the None to NaN when the column also holds dates.
        assert pd.isna(coverage.loc["X", "scored_from"])

    def test_scored_never_exceeds_available(self) -> None:
        coverage = crisis_coverage(self._panel(200), window=50)
        for crisis in CRISIS_WINDOWS:
            scored = int(coverage.loc["X", f"{crisis.tag}_scored"])
            available = int(coverage.loc["X", f"{crisis.tag}_available"])
            assert scored <= available
