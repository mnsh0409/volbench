"""Grid orchestration (`volbench.runner`, D-027).

Four properties, each with its own class:

- ``TestGridExpansion`` — the grid is a total order over
  ``(asset, model, horizon, arm)``, not whatever order the caller's tuples
  happened to be in.
- ``TestResumability`` — a re-run adds only what is missing and rewrites
  nothing. Checked on the *bytes* and on the mtimes, because "did not change"
  and "was not rewritten" are different claims and only the second one makes
  an interrupted grid safe to resume.
- ``TestFaultIsolation`` — a cell that raises costs one cell.
- ``TestSerialParallelIdentity`` — **the gate that matters**: the parallel
  backend produces byte-identical fragments to the serial one, which is
  D-011's H4 claim ("same forecasts, three execution paths, verified
  identity"). If this ever fails, the fix is upstream, never here.

Everything here uses real model adapters over a synthetic series: the point is
the orchestration, and a fake model would not exercise pickling across the
process boundary the way a real factory does.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from volbench.determinism import thread_pin
from volbench.dist import Distribution, Normal
from volbench.execute import Executor, ProcessExecutor, SerialExecutor
from volbench.models import EWMA, HAR, NaiveVol
from volbench.models.base import FitDiagnostics
from volbench.results import ResultsStore
from volbench.runner import (
    LANE_ORDER,
    AssetData,
    Cell,
    CellOutcome,
    GridSpec,
    MappingDataSource,
    ModelConfig,
    ProtocolArm,
    RunManifest,
    read_grid_results,
    run_grid,
)

WINDOW = 120
N_OBS = 320
N_ORIGINS = N_OBS - WINDOW  # step 1, horizon 1


# --------------------------------------------------------------------------
# data and factories — module level, because the process backend pickles them
# --------------------------------------------------------------------------


def make_asset(asset: str, seed: int, n: int = N_OBS) -> AssetData:
    """A synthetic asset on a real calendar: returns, a variance target, and
    the same series as the variance-fed models' fit input."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    variance = np.exp(rng.normal(np.log(1e-4), 0.35, n))
    returns = rng.normal(0.0, np.sqrt(variance))
    return AssetData(
        asset=asset,
        returns=pd.Series(returns, index=index),
        proxy=pd.Series(variance, index=index),
        proxy_name="synthetic_variance",
        variance=pd.Series(variance, index=index),
        data_spec={"fixture": "test_runner", "n": n},
    )


class ExplodingModel:
    """A model whose ``fit`` cannot even be reached: the factory itself raises.

    A failure *inside* fit is already isolated per origin by the evaluator, so
    to test the runner's own isolation the cell has to fail before any origin
    runs — which is what constructing the model does here.
    """

    def __init__(self) -> None:
        raise RuntimeError("this model was never going to work")


class FlakyFitted:
    """A fit that reports a status: a fallback when its window ends on a down day.

    Stands in for GARCH's real behaviour without GARCH's cost. The predicate is
    a function of the *window*, not of a call counter, because the runner
    constructs a fresh model per refit block — a counter would reset on every
    block at ``refit_every=1`` and quietly never fall back at all.
    """

    def __init__(self, fell_back: bool) -> None:
        self.fell_back = fell_back

    @property
    def name(self) -> str:
        return "flaky"

    def spec(self) -> dict[str, Any]:
        return {"model": "flaky"}

    def fit_diagnostics(self) -> FitDiagnostics:
        if self.fell_back:
            return FitDiagnostics(converged=False, fallback="ewma", detail="flag=9")
        return FitDiagnostics(converged=True, detail="flag=0")

    def predict(self, h: int) -> Distribution:
        return Normal(mu=0.0, sigma=0.01)


class Flaky:
    """Fits :class:`FlakyFitted`. Picklable; holds no state between fits."""

    @property
    def name(self) -> str:
        return "flaky"

    def spec(self) -> dict[str, Any]:
        return {"model": "flaky"}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FlakyFitted:
        return FlakyFitted(bool(train[-1] < 0.0))


class ConstantVol:
    """A trivially cheap model, for grids where the model is beside the point."""

    @property
    def name(self) -> str:
        return "constant_vol"

    def spec(self) -> dict[str, Any]:
        return {"model": "constant_vol", "sigma": 0.01}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> ConstantVol:
        return self

    def predict(self, h: int) -> Distribution:
        return Normal(mu=0.0, sigma=0.01)


class CountingNaive:
    """A NaiveVol that counts how often it is constructed and fitted.

    Class-level counters rather than a closure, because the factory has to stay
    picklable; only the serial backend reads them, since counters do not cross
    a process boundary.
    """

    constructions = 0
    fits = 0

    def __init__(self) -> None:
        type(self).constructions += 1
        self._inner = NaiveVol()

    @classmethod
    def reset(cls) -> None:
        cls.constructions = 0
        cls.fits = 0

    @property
    def name(self) -> str:
        return "counting_naive"

    def spec(self) -> dict[str, Any]:
        return {"model": "counting_naive"}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> Any:
        type(self).fits += 1
        return self._inner.fit(train)


def cpu_models() -> tuple[ModelConfig, ...]:
    return (
        ModelConfig("ewma", functools.partial(EWMA, lambda_=0.94)),
        ModelConfig("har", HAR, fits_on_variance=True),
        ModelConfig("naive", NaiveVol),
    )


@pytest.fixture(scope="module")
def data() -> dict[str, AssetData]:
    return {"AAA": make_asset("AAA", seed=3), "BBB": make_asset("BBB", seed=4)}


@pytest.fixture(scope="module")
def grid() -> GridSpec:
    return GridSpec(
        assets=("AAA", "BBB"),
        models=cpu_models(),
        horizons=(1,),
        arms=(ProtocolArm(label="w120", window=WINDOW, refit_every=21),),
        seed=7,
    )


# --------------------------------------------------------------------------
# expansion and ordering
# --------------------------------------------------------------------------


class TestGridExpansion:
    def test_size_is_the_full_cross(self, grid: GridSpec) -> None:
        assert grid.size == 2 * 3 * 1 * 1
        assert len(grid.cells()) == grid.size

    def test_cells_are_in_one_total_order_whatever_order_they_were_declared(self) -> None:
        """Two descriptions of one grid must expand identically, or two runs
        produce manifests nobody can diff."""
        forward = GridSpec(
            assets=("AAA", "BBB"),
            models=(ModelConfig("a", NaiveVol), ModelConfig("b", NaiveVol)),
            horizons=(1, 5),
            arms=(ProtocolArm("x", window=WINDOW), ProtocolArm("y", window=WINDOW)),
        )
        backward = GridSpec(
            assets=("BBB", "AAA"),
            models=(ModelConfig("b", NaiveVol), ModelConfig("a", NaiveVol)),
            horizons=(5, 1),
            arms=(ProtocolArm("y", window=WINDOW), ProtocolArm("x", window=WINDOW)),
        )
        assert [c.key for c in forward.cells()] == [c.key for c in backward.cells()]
        assert [c.key for c in forward.cells()] == sorted(c.key for c in forward.cells())

    def test_duplicate_labels_are_refused(self) -> None:
        with pytest.raises(ValueError, match="model labels must be distinct"):
            GridSpec(assets=("A",), models=(ModelConfig("m", NaiveVol),) * 2)
        with pytest.raises(ValueError, match="arm labels must be distinct"):
            GridSpec(
                assets=("A",),
                models=(ModelConfig("m", NaiveVol),),
                arms=(ProtocolArm("x"), ProtocolArm("x")),
            )
        with pytest.raises(ValueError, match="assets must be distinct"):
            GridSpec(assets=("A", "A"), models=(ModelConfig("m", NaiveVol),))

    def test_an_empty_dimension_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            GridSpec(assets=(), models=(ModelConfig("m", NaiveVol),))

    def test_the_splitter_is_the_arms_and_nothing_is_hand_rolled(self) -> None:
        """CLAUDE.md rule 1: indices come from RollingOriginSplitter only."""
        cell = Cell(
            asset="AAA",
            model=ModelConfig("naive", NaiveVol),
            horizon=5,
            arm=ProtocolArm("x", window=250, refit_every=21, step=2),
            seed=1,
        )
        splitter = cell.splitter()
        assert (splitter.window, splitter.horizon, splitter.step, splitter.refit_every) == (
            250,
            5,
            2,
            21,
        )

    def test_lane_must_be_declared_not_guessed(self) -> None:
        with pytest.raises(ValueError, match="lane"):
            ModelConfig("m", NaiveVol, lane="gpu0")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# a plain run
# --------------------------------------------------------------------------


class TestRun:
    def test_every_cell_is_scored_and_recorded_in_grid_order(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        store = ResultsStore(tmp_path / "store")
        manifest = run_grid(grid, data, store)

        assert [c.index for c in manifest.cells] == list(range(grid.size))
        assert [(c.asset, c.model, c.horizon, c.arm) for c in manifest.cells] == [
            c.key for c in grid.cells()
        ]
        assert manifest.n_computed == grid.size
        assert (manifest.n_cached, manifest.n_failed) == (0, 0)
        for outcome in manifest.cells:
            assert outcome.config_hash is not None
            assert outcome.n_rows == N_ORIGINS
            assert outcome.n_missing == 0
            assert store.has(outcome.config_hash)

    def test_distinct_cells_get_distinct_hashes(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        manifest = run_grid(grid, data, ResultsStore(tmp_path / "store"))
        hashes = [c.config_hash for c in manifest.cells]
        assert len(set(hashes)) == len(hashes)

    def test_a_mapping_is_accepted_as_a_data_source(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        explicit = run_grid(grid, MappingDataSource(data), ResultsStore(tmp_path / "a"))
        bare = run_grid(grid, data, ResultsStore(tmp_path / "b"))
        assert [c.config_hash for c in explicit.cells] == [c.config_hash for c in bare.cells]

    def test_the_source_is_read_once_per_asset_not_once_per_cell(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        """One read per asset is what makes every cell of that asset provably
        score against the same bytes — and it is also what stops a 200-cell
        grid re-parsing an archive 200 times."""
        calls: list[str] = []

        class Counting:
            def load(self, asset: str) -> AssetData:
                calls.append(asset)
                return data[asset]

        run_grid(grid, Counting(), ResultsStore(tmp_path / "store"))
        assert sorted(calls) == ["AAA", "BBB"]

    def test_a_source_failure_stops_the_run_rather_than_failing_every_cell(
        self, grid: GridSpec, tmp_path: Path
    ) -> None:
        """A wrong path is a wasted grid, not a result: it must surface as an
        exception, not as 6 identical 'failed' rows in a manifest."""

        class Broken:
            def load(self, asset: str) -> AssetData:
                raise FileNotFoundError(f"no archive for {asset}")

        with pytest.raises(FileNotFoundError):
            run_grid(grid, Broken(), ResultsStore(tmp_path / "store"))

    def test_running_without_a_store_still_scores_everything(
        self, grid: GridSpec, data: dict[str, AssetData]
    ) -> None:
        manifest = run_grid(grid, data, None)
        assert manifest.n_computed == grid.size
        assert all(c.config_hash is not None for c in manifest.cells)

    def test_read_grid_results_returns_this_grids_rows_only(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        store = ResultsStore(tmp_path / "store")
        manifest = run_grid(grid, data, store)
        # A cell from a *different* study, sharing the store.
        other = GridSpec(
            assets=("AAA",),
            models=(ModelConfig("constant", ConstantVol),),
            arms=(ProtocolArm("w120", window=WINDOW),),
            seed=7,
        )
        run_grid(other, data, store)

        rows = read_grid_results(store, manifest)
        assert len(rows) == grid.size * N_ORIGINS
        assert set(rows["config_hash"]) == {c.config_hash for c in manifest.cells}
        assert len(store.read_all()) > len(rows)


# --------------------------------------------------------------------------
# resumability
# --------------------------------------------------------------------------


def _fragment_state(store: ResultsStore) -> dict[str, tuple[bytes, int]]:
    return {
        h: (store.fragment_path(h).read_bytes(), store.fragment_path(h).stat().st_mtime_ns)
        for h in store.config_hashes()
    }


class TestResumability:
    def test_a_second_run_recomputes_nothing_and_rewrites_nothing(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        store = ResultsStore(tmp_path / "store")
        first = run_grid(grid, data, store)
        before = _fragment_state(store)

        second = run_grid(grid, data, store)

        assert second.n_cached == grid.size
        assert second.n_computed == 0
        assert [c.config_hash for c in second.cells] == [c.config_hash for c in first.cells]
        # Bytes AND mtimes: "unchanged" and "not rewritten" are different
        # claims, and only the second one makes an interrupted grid safe.
        assert _fragment_state(store) == before

    def test_an_interrupted_run_adds_only_the_missing_cells(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        store = ResultsStore(tmp_path / "store")
        complete = run_grid(grid, data, store)
        reference = _fragment_state(store)

        # Simulate the interruption: two cells never made it to disk.
        lost = [complete.cells[1].config_hash, complete.cells[4].config_hash]
        for h in lost:
            assert h is not None
            store.fragment_path(h).unlink()
            store.config_path(h).unlink()

        resumed = run_grid(grid, data, store)

        recomputed = {c.config_hash for c in resumed.cells if c.status == "computed"}
        assert recomputed == set(lost)
        assert resumed.n_cached == grid.size - 2
        after = _fragment_state(store)
        assert set(after) == set(reference)
        for h, (payload, mtime) in reference.items():
            assert after[h][0] == payload, h  # every fragment byte-identical
            if h not in lost:
                assert after[h][1] == mtime, h  # and the survivors untouched

    def test_a_cached_cell_fits_nothing_at_all(
        self, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        """The short-circuit has to happen before any work, or 'resumable'
        only means 'does not duplicate rows'. Counted rather than timed: a
        wall-clock comparison would pass for a cell that refitted quickly."""
        store = ResultsStore(tmp_path / "store")
        small = GridSpec(
            assets=("AAA",),
            models=(ModelConfig("counting", CountingNaive),),
            arms=(ProtocolArm("w120", window=WINDOW, refit_every=21),),
            seed=7,
        )
        CountingNaive.reset()
        first = run_grid(small, data, store)
        assert first.n_computed == 1
        fits_for_a_real_run = CountingNaive.fits
        assert fits_for_a_real_run > 0

        CountingNaive.reset()
        second = run_grid(small, data, store)
        assert second.n_cached == 1
        assert CountingNaive.fits == 0
        # The probe run_backtest makes to read name/spec() for the hash is the
        # only construction a cached cell is allowed, and it never fits.
        assert CountingNaive.constructions == 1

    def test_overwrite_replaces_instead_of_skipping(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        store = ResultsStore(tmp_path / "store")
        run_grid(grid, data, store)
        before = _fragment_state(store)

        again = run_grid(grid, data, store, overwrite=True)

        assert again.n_computed == grid.size
        after = _fragment_state(store)
        assert {h: payload for h, (payload, _) in after.items()} == {
            h: payload for h, (payload, _) in before.items()
        }


# --------------------------------------------------------------------------
# fault isolation
# --------------------------------------------------------------------------


class TestFaultIsolation:
    def test_a_failing_cell_is_recorded_and_the_grid_continues(
        self, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        grid = GridSpec(
            assets=("AAA", "BBB"),
            models=(ModelConfig("boom", ExplodingModel), ModelConfig("naive", NaiveVol)),
            arms=(ProtocolArm("w120", window=WINDOW),),
            seed=7,
        )
        manifest = run_grid(grid, data, ResultsStore(tmp_path / "store"))

        failed = manifest.failures
        assert {c.model for c in failed} == {"boom"}
        assert len(failed) == 2
        for outcome in failed:
            assert outcome.error is not None
            assert outcome.error.startswith("RuntimeError: this model was never going to work")
            assert outcome.config_hash is None
            assert outcome.n_rows == 0
        # The other model's cells are untouched by their neighbour's failure.
        good = [c for c in manifest.cells if c.model == "naive"]
        assert len(good) == 2
        assert all(c.status == "computed" and c.n_rows == N_ORIGINS for c in good)

    def test_a_missing_variance_series_fails_only_the_cells_that_need_one(
        self, tmp_path: Path
    ) -> None:
        """A return-fed model on the same asset must be unaffected."""
        returns_only = make_asset("CCC", seed=9)
        without = AssetData(
            asset="CCC",
            returns=returns_only.returns,
            proxy=returns_only.proxy,
            proxy_name=returns_only.proxy_name,
            variance=None,
        )
        grid = GridSpec(
            assets=("CCC",),
            models=(ModelConfig("har", HAR, fits_on_variance=True), ModelConfig("naive", NaiveVol)),
            arms=(ProtocolArm("w120", window=WINDOW),),
            seed=7,
        )
        manifest = run_grid(grid, {"CCC": without}, ResultsStore(tmp_path / "store"))

        by_model = {c.model: c for c in manifest.cells}
        assert by_model["har"].status == "failed"
        assert "variance series" in (by_model["har"].error or "")
        assert by_model["naive"].status == "computed"

    def test_per_origin_failures_stay_visible_as_n_missing(self, tmp_path: Path) -> None:
        """A cell that "succeeded" while scoring nothing must not look clean
        from the manifest: that is how a broken model gets into a table."""
        asset = make_asset("DDD", seed=11)
        broken_proxy = asset.proxy.copy()
        broken_proxy.iloc[:] = np.nan
        grid = GridSpec(
            assets=("DDD",),
            models=(ModelConfig("naive", NaiveVol),),
            arms=(ProtocolArm("w120", window=WINDOW),),
            seed=7,
        )
        manifest = run_grid(
            grid,
            {
                "DDD": AssetData(
                    asset="DDD",
                    returns=asset.returns,
                    proxy=broken_proxy,
                    proxy_name="all_nan",
                )
            },
            ResultsStore(tmp_path / "store"),
        )
        outcome = manifest.cells[0]
        assert outcome.status == "computed"
        assert outcome.n_missing == outcome.n_rows == N_ORIGINS


# --------------------------------------------------------------------------
# lanes
# --------------------------------------------------------------------------


class RecordingExecutor:
    """Wraps a real executor and records the order lanes were dispatched in."""

    def __init__(self, tag: str, log: list[tuple[str, int]]) -> None:
        self.tag, self.log, self.inner = tag, log, SerialExecutor()

    def map(self, fn: Any, items: Any) -> list[Any]:
        materialized = list(items)
        self.log.append((self.tag, len(materialized)))
        return self.inner.map(fn, materialized)


class TestLanes:
    def test_gpu_configs_go_to_the_gpu_executor_and_cpu_configs_do_not(
        self, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        log: list[tuple[str, int]] = []
        grid = GridSpec(
            assets=("AAA",),
            models=(
                ModelConfig("naive", NaiveVol),
                ModelConfig("constant_gpu", ConstantVol, lane="gpu"),
                ModelConfig("ewma", functools.partial(EWMA, lambda_=0.94)),
            ),
            arms=(ProtocolArm("w120", window=WINDOW),),
            seed=7,
        )
        manifest = run_grid(
            grid,
            data,
            ResultsStore(tmp_path / "store"),
            cpu_executor=RecordingExecutor("cpu", log),
            gpu_executor=RecordingExecutor("gpu", log),
        )
        assert log == [("cpu", 2), ("gpu", 1)]
        assert {c.model: c.lane for c in manifest.cells} == {
            "constant_gpu": "gpu",
            "ewma": "cpu",
            "naive": "cpu",
        }

    def test_the_cpu_lane_runs_first(self) -> None:
        """Not cosmetic: the CPU lane forks, the GPU lane initializes CUDA, and
        forking after a CUDA context exists is undefined behaviour."""
        assert LANE_ORDER == ("cpu", "gpu")

    def test_an_empty_lane_is_not_dispatched_at_all(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        log: list[tuple[str, int]] = []
        run_grid(
            grid,
            data,
            ResultsStore(tmp_path / "store"),
            cpu_executor=RecordingExecutor("cpu", log),
            gpu_executor=RecordingExecutor("gpu", log),
        )
        assert log == [("cpu", grid.size)]

    def test_the_lane_is_never_inferred_from_a_models_name(self) -> None:
        """D-027: routing is declared. A model called 'patchtst' is on the CPU
        lane unless someone says otherwise, and that must stay true — a
        name-based guess misroutes the first adapter that breaks the pattern."""
        assert ModelConfig("patchtst", NaiveVol).lane == "cpu"
        assert ModelConfig("chronos_gpu_tsfm", NaiveVol).lane == "cpu"


# --------------------------------------------------------------------------
# THE GATE — D-011 H4
# --------------------------------------------------------------------------


class TestSerialParallelIdentity:
    """Same forecasts, two execution paths, verified identity (D-011).

    This is the claim the whole ``execute`` seam exists to support, and it is
    checked on the parquet bytes rather than on the frames: a difference the
    dtypes or the row order absorbed would still be a difference in what a
    later analysis reads off disk.

    If this test fails, the fix is in whatever the parallel path did
    differently — never in this assertion.
    """

    def test_the_parallel_backend_produces_byte_identical_fragments(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        serial_store = ResultsStore(tmp_path / "serial")
        parallel_store = ResultsStore(tmp_path / "parallel")

        serial = run_grid(grid, data, serial_store, cpu_executor=SerialExecutor())
        parallel = run_grid(
            grid, data, parallel_store, cpu_executor=ProcessExecutor(workers=3)
        )

        assert [c.config_hash for c in parallel.cells] == [c.config_hash for c in serial.cells]
        assert serial.n_computed == parallel.n_computed == grid.size
        assert serial_store.config_hashes() == parallel_store.config_hashes()
        for digest in serial_store.config_hashes():
            assert (
                parallel_store.fragment_path(digest).read_bytes()
                == serial_store.fragment_path(digest).read_bytes()
            ), digest

    def test_the_identity_check_can_actually_fail(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        """An inert-proof companion: the comparison above compares something.
        Perturb one input by one ulp and the fragments must differ."""
        a = ResultsStore(tmp_path / "a")
        b = ResultsStore(tmp_path / "b")
        run_grid(grid, data, a, cpu_executor=SerialExecutor())

        nudged = dict(data)
        original = data["AAA"]
        moved = original.returns.copy()
        moved.iloc[0] = np.nextafter(moved.iloc[0], np.inf)
        nudged["AAA"] = AssetData(
            asset=original.asset,
            returns=moved,
            proxy=original.proxy,
            proxy_name=original.proxy_name,
            variance=original.variance,
            data_spec=original.data_spec,
        )
        run_grid(grid, nudged, b, cpu_executor=ProcessExecutor(workers=2))

        # The content digest moved, so the AAA cells are different experiments
        # entirely — which is the cache-identity control doing its job.
        assert set(a.config_hashes()) != set(b.config_hashes())

    def test_the_parallel_backend_isolates_faults_the_same_way(
        self, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        grid = GridSpec(
            assets=("AAA", "BBB"),
            models=(ModelConfig("boom", ExplodingModel), ModelConfig("naive", NaiveVol)),
            arms=(ProtocolArm("w120", window=WINDOW),),
            seed=7,
        )
        manifest = run_grid(
            grid,
            data,
            ResultsStore(tmp_path / "store"),
            cpu_executor=ProcessExecutor(workers=2),
        )
        assert manifest.n_failed == 2
        assert all("RuntimeError" in (c.error or "") for c in manifest.failures)
        assert manifest.n_computed == 2

    def test_the_grid_ordering_is_the_same_under_either_backend(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        """The pool returns work in whatever order it finishes; the manifest
        must not."""
        serial = run_grid(grid, data, ResultsStore(tmp_path / "s"))
        parallel = run_grid(
            grid, data, ResultsStore(tmp_path / "p"), cpu_executor=ProcessExecutor(workers=3)
        )
        assert [(c.index, c.asset, c.model, c.horizon, c.arm) for c in parallel.cells] == [
            (c.index, c.asset, c.model, c.horizon, c.arm) for c in serial.cells
        ]


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


class TestManifest:
    def test_it_records_every_cell_with_hash_status_and_wall_clock(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        path = tmp_path / "store" / "manifest.json"
        manifest = run_grid(
            grid, data, ResultsStore(tmp_path / "store"), manifest_path=path
        )
        loaded = json.loads(path.read_text(encoding="utf-8"))

        assert loaded["n_cells"] == grid.size
        assert loaded["n_computed"] == grid.size
        assert [c["index"] for c in loaded["cells"]] == list(range(grid.size))
        for record, outcome in zip(loaded["cells"], manifest.cells, strict=True):
            assert record["config_hash"] == outcome.config_hash
            assert record["status"] == outcome.status
            assert record["wall_clock_s"] >= 0.0
            assert set(record) == {
                "index",
                "asset",
                "model",
                "horizon",
                "arm",
                "lane",
                "status",
                "config_hash",
                "n_rows",
                "n_missing",
                "n_fits",
                "n_fits_fallback",
                "n_fits_nonconverged",
                "wall_clock_s",
                "error",
            }

    def test_two_runs_of_one_grid_agree_on_everything_except_timings(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        first = run_grid(grid, data, ResultsStore(tmp_path / "a"))
        second = run_grid(grid, data, ResultsStore(tmp_path / "b"))

        def without_timings(manifest: RunManifest) -> list[dict[str, Any]]:
            return [
                {k: v for k, v in c.as_json().items() if k != "wall_clock_s"}
                for c in manifest.cells
            ]

        assert without_timings(first) == without_timings(second)

    def test_to_frame_is_one_row_per_cell(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        manifest = run_grid(grid, data, ResultsStore(tmp_path / "store"))
        frame = manifest.to_frame()
        assert len(frame) == grid.size
        assert frame["index"].tolist() == list(range(grid.size))

    def test_config_hashes_are_addressable_by_cell_key(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        manifest = run_grid(grid, data, ResultsStore(tmp_path / "store"))
        assert set(manifest.config_hashes) == {c.key for c in grid.cells()}

    def test_on_cell_sees_every_outcome(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        seen: list[CellOutcome] = []
        manifest = run_grid(
            grid, data, ResultsStore(tmp_path / "store"), on_cell=seen.append
        )
        assert sorted(o.index for o in seen) == [c.index for c in manifest.cells]

    def test_a_manifest_is_never_read_back_as_run_state(self, tmp_path: Path) -> None:
        """Resuming reads the store, not a manifest, so a stale or hand-edited
        manifest cannot corrupt a grid. Pinned as an interface fact: there is
        no reader."""
        assert not hasattr(RunManifest, "read")
        assert not hasattr(RunManifest, "load")


class TestLeakageCanary:
    """The leakage-check skill's demanded canary, at the *grid* level.

    Corrupt every observation strictly after a cutoff T, run the same grid
    again, and require every forecast whose target lands at or before T to be
    bit-identical. The runner adds no index arithmetic of its own — that is
    the design — so this is a check that it did not introduce any: a cell that
    read its data through anything but the splitter's own windows, or a
    parallel backend that pooled state across cells, would move these numbers.

    It runs under both backends for that second reason.
    """

    CUTOFF = WINDOW + 40

    @staticmethod
    def _corrupt(source: AssetData, start: int, stop: int | None, seed: int) -> AssetData:
        rng = np.random.default_rng(seed)
        returns = source.returns.copy()
        n = returns.iloc[start:stop].size
        returns.iloc[start:stop] = rng.normal(0.0, 0.5, size=n)
        variance = source.proxy.copy()
        variance.iloc[start:stop] = np.exp(rng.normal(0.0, 1.0, size=n))
        return AssetData(
            asset=source.asset,
            returns=returns,
            proxy=variance,
            proxy_name=source.proxy_name,
            variance=variance.copy(),
            data_spec=source.data_spec,
        )

    def _rows(
        self, grid: GridSpec, asset: AssetData, store: ResultsStore, executor: Executor
    ) -> pd.DataFrame:
        manifest = run_grid(grid, {asset.asset: asset}, store, cpu_executor=executor)
        assert manifest.n_failed == 0, manifest.failures
        rows = read_grid_results(store, manifest)
        # `config_hash` legitimately differs — the data digest changed — so
        # compare the scored numbers, not the provenance columns.
        scores = [c for c in rows.columns if c != "config_hash"]
        return (
            rows[rows["target_index"] <= self.CUTOFF][scores]
            .sort_values(["model", "origin_index", "horizon"], kind="stable")
            .reset_index(drop=True)
        )

    @pytest.mark.parametrize(
        "executor",
        [
            pytest.param(SerialExecutor(), id="serial"),
            pytest.param(ProcessExecutor(workers=2), id="process"),
        ],
    )
    def test_future_corruption_cannot_change_past_forecasts(
        self, data: dict[str, AssetData], tmp_path: Path, executor: Executor
    ) -> None:
        grid = GridSpec(
            assets=("AAA",),
            models=cpu_models(),
            arms=(ProtocolArm("w120", window=WINDOW),),
            seed=7,
        )
        clean_asset = data["AAA"]
        dirty_asset = self._corrupt(clean_asset, self.CUTOFF + 1, None, seed=999)

        clean = self._rows(grid, clean_asset, ResultsStore(tmp_path / "clean"), executor)
        dirty = self._rows(grid, dirty_asset, ResultsStore(tmp_path / "dirty"), executor)

        assert len(clean) > 0, "cutoff left no fully-clean origins to compare"
        pd.testing.assert_frame_equal(clean, dirty, check_exact=True)

    def test_the_canary_can_actually_fail(
        self, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        """Corrupting the *past* must move the very numbers the future did
        not, or the comparison above is comparing nothing."""
        grid = GridSpec(
            assets=("AAA",),
            models=cpu_models(),
            arms=(ProtocolArm("w120", window=WINDOW),),
            seed=7,
        )
        clean_asset = data["AAA"]
        dirty_asset = self._corrupt(clean_asset, 0, self.CUTOFF + 1, seed=1234)

        clean = self._rows(grid, clean_asset, ResultsStore(tmp_path / "clean"), SerialExecutor())
        dirty = self._rows(grid, dirty_asset, ResultsStore(tmp_path / "dirty"), SerialExecutor())

        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(clean, dirty, check_exact=True)


def test_the_runner_only_ever_hands_a_cell_a_serial_executor() -> None:
    """execute.py rule 1, read off the source rather than trusted: a cell
    running inside a pool must not open a pool of its own."""
    import inspect

    from volbench import runner

    source = inspect.getsource(runner._run_cell)
    assert "executor=SerialExecutor()" in source
    assert isinstance(SerialExecutor(), Executor)


# --------------------------------------------------------------------------
# D-032: a fallback is counted, and the machine is on the record
# --------------------------------------------------------------------------


class TestFitCountsReachTheManifest:
    """A cell that quietly ran a different estimator on some origins has to be
    readable from the manifest, without opening a parquet."""

    @pytest.fixture
    def flaky_grid(self) -> GridSpec:
        return GridSpec(
            assets=("AAA",),
            models=(ModelConfig("flaky", Flaky),),
            arms=(ProtocolArm(label="h", window=WINDOW, refit_every=1),),
        )

    def test_it_counts_fallbacks_per_scheduled_fit(
        self, flaky_grid: GridSpec, tmp_path: Path
    ) -> None:
        """Checked against the rows themselves rather than against a number
        computed twice the same way: the claim is that the manifest count *is*
        what the fragment says."""
        data = {"AAA": make_asset("AAA", seed=1)}
        store = ResultsStore(tmp_path)
        manifest = run_grid(flaky_grid, data, store)
        (cell,) = manifest.cells

        assert cell.config_hash is not None
        rows = store.read(cell.config_hash)
        per_fit = rows.groupby("fit_origin")["fit_status"].first()
        expected = int(per_fit.str.startswith("fallback=").sum())

        assert cell.n_fits == N_ORIGINS == len(per_fit)
        assert cell.n_fits_fallback == expected
        assert 0 < expected < cell.n_fits, "the fixture must exercise both branches"
        assert cell.n_fits_nonconverged == cell.n_fits_fallback
        assert cell.fallback_rate == pytest.approx(expected / cell.n_fits)
        assert manifest.fallback_cells == (cell,)

    def test_a_model_that_reports_nothing_gets_a_nan_rate_not_a_zero(
        self, tmp_path: Path
    ) -> None:
        """"No fit fell back" and "no fit said" are different claims, and only
        one of them is evidence."""
        grid = GridSpec(
            assets=("AAA",),
            models=(ModelConfig("const", ConstantVol),),
            arms=(ProtocolArm(label="h", window=WINDOW, refit_every=1),),
        )
        manifest = run_grid(grid, {"AAA": make_asset("AAA", seed=1)}, ResultsStore(tmp_path))
        (cell,) = manifest.cells
        assert cell.n_fits == 0
        assert np.isnan(cell.fallback_rate)
        assert manifest.fallback_cells == ()

    def test_the_counts_are_per_fit_not_per_row(self, tmp_path: Path) -> None:
        """At ``refit_every=7`` one fit serves seven origins. Counting rows
        would report a rate that is really a row-weighting of the refit
        cadence."""
        grid = GridSpec(
            assets=("AAA",),
            models=(ModelConfig("flaky", Flaky),),
            arms=(ProtocolArm(label="h", window=WINDOW, refit_every=7, recondition="none"),),
        )
        manifest = run_grid(grid, {"AAA": make_asset("AAA", seed=1)}, ResultsStore(tmp_path))
        (cell,) = manifest.cells
        assert cell.n_rows == N_ORIGINS
        assert cell.n_fits < cell.n_rows
        assert cell.n_fits == len(range(0, N_ORIGINS, 7))

    def test_a_resumed_cell_reports_the_same_counts_it_was_filled_with(
        self, flaky_grid: GridSpec, tmp_path: Path
    ) -> None:
        """Counted from the fragment, so resuming a grid does not silently
        reset its fallback rate to zero."""
        data = {"AAA": make_asset("AAA", seed=1)}
        store = ResultsStore(tmp_path)
        first = run_grid(flaky_grid, data, store)
        second = run_grid(flaky_grid, data, store)
        assert second.cells[0].status == "cached"
        assert second.cells[0].n_fits == first.cells[0].n_fits
        assert second.cells[0].n_fits_fallback == first.cells[0].n_fits_fallback

    def test_a_failed_cell_claims_no_fits(self, tmp_path: Path) -> None:
        grid = GridSpec(
            assets=("AAA",),
            models=(ModelConfig("boom", ExplodingModel),),
            arms=(ProtocolArm(label="h", window=WINDOW),),
        )
        manifest = run_grid(grid, {"AAA": make_asset("AAA", seed=1)}, ResultsStore(tmp_path))
        (cell,) = manifest.cells
        assert cell.status == "failed"
        assert (cell.n_fits, cell.n_fits_fallback) == (0, 0)


class TestManifestRecordsTheMachine:
    def test_it_carries_the_thread_count_and_the_blas_build(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        """D-032: the hash carries only ``blas_threads``; the manifest says
        more, because a reader diagnosing two runs that disagree needs the BLAS
        build and the pins as they stood."""
        manifest = run_grid(grid, data, ResultsStore(tmp_path))
        environment = manifest.environment
        assert environment["blas_threads"] == thread_pin()
        assert "name" in environment["blas"]
        assert "kernel_signature" in environment

    def test_the_environment_survives_the_json_round_trip(
        self, grid: GridSpec, data: dict[str, AssetData], tmp_path: Path
    ) -> None:
        path = tmp_path / "manifest.json"
        run_grid(grid, data, ResultsStore(tmp_path / "store"), manifest_path=path)
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["environment"]["blas_threads"] == thread_pin()
        assert written["n_fits"] >= 0

    def test_a_hand_built_manifest_needs_no_environment(self) -> None:
        """It is a report, not an input; a test must be able to build one."""
        assert RunManifest(cells=()).environment == {}
