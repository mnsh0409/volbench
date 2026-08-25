"""Grid orchestration: run every cell of a study, resumably, and say what happened.

:func:`volbench.evaluate.run_backtest` scores **one** cell. This module runs
*the grid* — the full cross of ``asset x model-config x horizon x protocol-arm``
that docs/research_design.md describes — and is the last piece of the
evaluation stack that the paper's numbers go through.

It is deliberately thin. Everything that decides a *number* already lives
elsewhere and is not re-implemented here: indices come from
``RollingOriginSplitter``, identity from :func:`volbench.results.config_hash`,
persistence from :class:`~volbench.results.ResultsStore`, and *where* work runs
from :mod:`volbench.execute`. What this module adds is three properties a bare
loop over ``run_backtest`` would not have.

Resumable by construction
=========================
A cell is skipped when its ``config_hash`` is already in the store — the same
content-addressed short-circuit ``run_backtest`` already performs, surfaced in
the manifest as ``status="cached"``. Nothing is recomputed, no fragment is
rewritten, and the store is append-only, so an interrupted run resumes by being
run again: it adds exactly the missing cells and leaves every existing fragment
byte-identical. There is no run state on disk beyond the results themselves,
which is what makes "resume" and "extend the grid" the same operation.

Per-cell fault isolation
========================
Same philosophy as the evaluator's per-origin isolation (M1 report §4.5), one
level up: a cell that raises is recorded as ``status="failed"`` with the
exception's type and message, and the grid continues. A model that cannot fit
one asset must not cost the other 200 cells. As in the evaluator, only
``Exception`` is caught — ``KeyboardInterrupt`` and ``SystemExit`` propagate,
because a user ending a run is not a bad cell.

Note where the two levels meet: an exception *inside* a cell's origins never
reaches here at all — ``run_backtest`` has already turned it into a NaN row
with a ``missing_reason``. A cell recorded as failed here is one whose whole
setup failed (its series could not be loaded, its inputs are not on one
calendar, the store could not be written). ``n_missing`` on a *successful*
outcome is how the finer-grained failures stay visible from the manifest.

Deterministic ordering
======================
Cells are expanded and executed in a total order over
``(asset, model label, horizon, arm label)`` — plain strings and ints, never
set or dict iteration order — so two runs of the same grid produce manifests
that line up row for row and can be diffed. Wall-clock is the one field that
legitimately differs between two runs of one grid; everything else in the
manifest is a function of the grid and the store.

Lanes — the GPU is a resource, and the config says which cells need it
=====================================================================
Two TSFM or PatchTST cells running concurrently on one GPU do not merely
contend, they can exhaust its memory and take each other down. So the grid
has two *lanes* (D-027):

- ``lane="cpu"`` — fan out across processes (:class:`ProcessExecutor`);
- ``lane="gpu"`` — a serialized lane, run through its own executor, which is
  a :class:`SerialExecutor` by default (one cell at a time, in this process).

The lane is a field on :class:`ModelConfig`. It is **never** inferred from a
model's name: a name-based guess would silently misroute the first adapter
whose name does not match the pattern, and the failure mode is a CUDA OOM
halfway through a grid rather than an error anyone can read.

The two lanes run **in sequence, CPU first**, and that order is not
arbitrary. The CPU lane's default backend forks; the GPU lane initializes CUDA
in whichever process runs it. Forking a process that has already initialized a
CUDA context is undefined behaviour, so the fan-out has to happen before the
GPU is ever touched. Running the lanes concurrently would need the GPU lane in
its own process from the start (pass ``gpu_executor=ProcessExecutor(workers=1)``
to get that); it is not the default because the sequence above is the one whose
safety does not depend on what a model's ``fit`` happens to import.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol, runtime_checkable

import pandas as pd

from volbench.compaction import DEFAULT_INVALID_TARGET_POLICY, FitSeries, InvalidTargetPolicy
from volbench.evaluate import DEFAULT_LEVELS, ModelFactory, Recondition, run_backtest
from volbench.execute import Executor, SerialExecutor
from volbench.results import ResultsStore
from volbench.splitter import RollingOriginSplitter

__all__ = [
    "LANE_ORDER",
    "AssetData",
    "Cell",
    "CellOutcome",
    "CellStatus",
    "DataSource",
    "GridSpec",
    "Lane",
    "MappingDataSource",
    "ModelConfig",
    "ProtocolArm",
    "RunManifest",
    "read_grid_results",
    "run_grid",
]

logger = logging.getLogger(__name__)

#: Which execution lane a model config belongs to. See the module docstring.
Lane = Literal["cpu", "gpu"]

#: The order lanes are run in. CPU first so the fan-out forks before anything
#: initializes CUDA; a tuple, not a set, because the order is the point.
LANE_ORDER: Final[tuple[Lane, ...]] = ("cpu", "gpu")

#: What became of one cell. ``"cached"`` means its fragment was already in the
#: store and nothing was fitted; ``"failed"`` means the cell itself raised.
CellStatus = Literal["computed", "cached", "failed"]


# --------------------------------------------------------------------------
# grid description
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """One model configuration in the grid, and how it must be run.

    ``factory`` is the zero-argument callable ``run_backtest`` takes: a
    module-level class or a :func:`functools.partial` over one, never a lambda
    or a closure — the process backend pickles it across a boundary.

    ``fits_on_variance`` routes the realized-variance series to the model
    instead of the returns (HAR, the log-RV models, PatchTST). Getting it wrong
    feeds a model the wrong units rather than raising, which is why it is
    declared per config and not guessed.

    ``lane`` is the GPU-contention control (module docstring, D-027). Explicit,
    never derived from ``label`` or from the model's own ``name``.
    """

    label: str
    factory: ModelFactory
    fits_on_variance: bool = False
    lane: Lane = "cpu"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("model label must not be empty")
        if self.lane not in ("cpu", "gpu"):
            raise ValueError(f"lane must be 'cpu' or 'gpu', got {self.lane!r}")


@dataclass(frozen=True)
class ProtocolArm:
    """One protocol arm: the settings that are varied *as an experiment*.

    Everything here reaches the config hash — through the splitter's own fields
    (``window``, ``step``, ``refit_every``) or through ``protocol``
    (``recondition``, ``invalid_target_policy``) — so two arms can never share a
    cached cell. ``label`` is the study's handle for the arm and is what the
    manifest is keyed by; it is *not* hashed, because the settings themselves
    already are, and a renamed arm is the same experiment.

    Defaults are the study's headline protocol: D-019's 500-observation window,
    daily re-conditioning between the 21-day refits (D-015), and D-018's
    compaction policy for invalid target days.
    """

    label: str
    window: int = 500
    refit_every: int = 21
    step: int = 1
    recondition: Recondition = "daily"
    #: Binds only on cells that are fed a variance series; a return-fed cell has
    #: no fit series to compact, and ``run_backtest`` records the policy exactly
    #: when it can change a number.
    invalid_target_policy: InvalidTargetPolicy = DEFAULT_INVALID_TARGET_POLICY

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("protocol arm label must not be empty")
        if self.recondition not in ("daily", "none"):
            raise ValueError(f"recondition must be 'daily' or 'none', got {self.recondition!r}")

    def splitter(self, horizon: int) -> RollingOriginSplitter:
        """The arm's splitter at ``horizon`` — the only source of indices."""
        return RollingOriginSplitter(
            window=self.window,
            horizon=horizon,
            step=self.step,
            refit_every=self.refit_every,
        )


@dataclass(frozen=True)
class Cell:
    """One fully-expanded grid cell. Picklable; carries no data.

    A cell is the unit :mod:`volbench.execute` distributes: it is scored
    independently of every other cell and merges into the store by
    ``config_hash`` alone.
    """

    asset: str
    model: ModelConfig
    horizon: int
    arm: ProtocolArm
    seed: int
    levels: tuple[float, ...] = DEFAULT_LEVELS

    @property
    def key(self) -> tuple[str, str, int, str]:
        """The cell's identity in the grid, and its sort key."""
        return (self.asset, self.model.label, self.horizon, self.arm.label)

    @property
    def lane(self) -> Lane:
        return self.model.lane

    def splitter(self) -> RollingOriginSplitter:
        return self.arm.splitter(self.horizon)

    def __str__(self) -> str:
        return f"{self.asset}/{self.model.label}/h{self.horizon}/{self.arm.label}"


@dataclass(frozen=True)
class GridSpec:
    """The cross to run: assets x model configs x horizons x protocol arms.

    ``horizons`` is the *splitter's* horizon, so a cell at horizon ``H`` scores
    every ``h`` in ``1..H`` and has its own origin set (the last origin must
    leave ``H`` observations after it). Cells at horizon 1 and horizon 5 are
    therefore different experiments, not a subset relation.

    ``seed`` and ``levels`` are grid-wide: a seed that varied per cell would
    make "the same experiment" undefined, and levels decide the output columns.
    """

    assets: tuple[str, ...]
    models: tuple[ModelConfig, ...]
    horizons: tuple[int, ...] = (1,)
    arms: tuple[ProtocolArm, ...] = (ProtocolArm(label="headline"),)
    seed: int = 20260825
    levels: tuple[float, ...] = DEFAULT_LEVELS

    def __post_init__(self) -> None:
        _require_distinct("assets", self.assets)
        _require_distinct("model labels", tuple(m.label for m in self.models))
        _require_distinct("arm labels", tuple(a.label for a in self.arms))
        _require_distinct("horizons", self.horizons)
        if not self.assets or not self.models or not self.horizons or not self.arms:
            raise ValueError("a grid needs at least one asset, model, horizon and arm")
        if any(h < 1 for h in self.horizons):
            raise ValueError(f"horizons must be >= 1, got {self.horizons}")

    @property
    def size(self) -> int:
        return len(self.assets) * len(self.models) * len(self.horizons) * len(self.arms)

    def cells(self) -> list[Cell]:
        """Every cell, in the grid's total order.

        Sorted by ``(asset, model label, horizon, arm label)``. The sort is what
        makes the ordering a property of the *grid* rather than of the order the
        caller happened to build its tuples in, so two descriptions of one grid
        produce one manifest ordering.
        """
        models = {m.label: m for m in self.models}
        arms = {a.label: a for a in self.arms}
        cells = [
            Cell(
                asset=asset,
                model=models[model_label],
                horizon=horizon,
                arm=arms[arm_label],
                seed=self.seed,
                levels=tuple(float(level) for level in self.levels),
            )
            for asset in sorted(self.assets)
            for model_label in sorted(models)
            for horizon in sorted(self.horizons)
            for arm_label in sorted(arms)
        ]
        return cells


def _require_distinct(what: str, values: Sequence[object]) -> None:
    if len(set(values)) != len(values):
        duplicated = sorted({str(v) for v in values if list(values).count(v) > 1})
        raise ValueError(f"{what} must be distinct; repeated: {duplicated}")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AssetData:
    """One asset's inputs, already on one calendar. ``eq=False``: holds pandas.

    ``returns`` and ``proxy`` are what ``run_backtest`` scores against;
    ``variance`` is the fit input for the variance-fed models and may be
    ``None`` for an asset no such model is run on. All three must share one
    index — ``run_backtest`` checks it, and checking it is the whole reason
    these travel together in one object rather than as three arguments.

    ``proxy_name`` is the run's scoring target (D-017: a property of the cell,
    never of the model), and ``data_spec`` is the provenance that joins the
    content digests in the config hash.
    """

    asset: str
    returns: pd.Series
    proxy: pd.Series
    proxy_name: str
    variance: pd.Series | None = None
    data_spec: Mapping[str, Any] = field(default_factory=dict)

    def fit_series(self, policy: InvalidTargetPolicy) -> FitSeries:
        """The variance fit input under ``policy`` (D-018).

        Raises if this asset carries no variance series: a variance-fed model
        with nothing to fit on is a configuration error, and it is reported as
        the failure of the cells that need it rather than silently substituting
        the returns — which would fit a variance model on returns and produce
        numbers nobody could interpret.
        """
        if self.variance is None:
            raise ValueError(
                f"{self.asset}: a variance-fed model needs a variance series, but this "
                "AssetData carries none (variance=None)"
            )
        return FitSeries.of(self.variance, policy=policy)


@runtime_checkable
class DataSource(Protocol):
    """Where the runner gets an asset's inputs.

    Implementations are called **once per asset in the parent process**, before
    any cell runs, so a series is read and digested exactly once per grid and
    every cell of that asset provably sees the same bytes.
    """

    def load(self, asset: str) -> AssetData:
        """The inputs for ``asset``."""
        ...


@dataclass(frozen=True, eq=False)
class MappingDataSource:
    """A :class:`DataSource` over an in-memory mapping. ``eq=False``: pandas."""

    data: Mapping[str, AssetData]

    def load(self, asset: str) -> AssetData:
        if asset not in self.data:
            raise KeyError(f"no data for asset {asset!r}; have {sorted(self.data)}")
        return self.data[asset]


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellOutcome:
    """What happened to one cell. One row of the run manifest.

    ``index`` is the cell's position in the grid's total order, so a manifest
    can be sorted back into that order whatever order the executors returned
    things in.

    ``wall_clock_s`` is the one field that legitimately differs between two runs
    of the same grid, and for a cached cell it is the cost of the store lookup,
    not of the work that was skipped.
    """

    index: int
    asset: str
    model: str
    horizon: int
    arm: str
    lane: Lane
    status: CellStatus
    config_hash: str | None
    n_rows: int
    #: Rows carrying a ``missing_reason`` — the evaluator's per-origin failures,
    #: kept visible at grid level so a "successful" cell that scored nothing
    #: cannot hide behind its status.
    n_missing: int
    wall_clock_s: float
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "asset": self.asset,
            "model": self.model,
            "horizon": self.horizon,
            "arm": self.arm,
            "lane": self.lane,
            "status": self.status,
            "config_hash": self.config_hash,
            "n_rows": self.n_rows,
            "n_missing": self.n_missing,
            "wall_clock_s": round(self.wall_clock_s, 6),
            "error": self.error,
        }


@dataclass(frozen=True)
class RunManifest:
    """Every cell the run attempted, in the grid's order.

    This is the run's own record: what was asked for, what was already there,
    what failed and why. It is not an input to anything — resuming reads the
    *store*, never a manifest — so a lost or stale manifest can never corrupt a
    grid, and two manifests of one grid differ only in their timings.
    """

    cells: tuple[CellOutcome, ...]

    @property
    def n_computed(self) -> int:
        return sum(1 for c in self.cells if c.status == "computed")

    @property
    def n_cached(self) -> int:
        return sum(1 for c in self.cells if c.status == "cached")

    @property
    def failures(self) -> tuple[CellOutcome, ...]:
        return tuple(c for c in self.cells if c.status == "failed")

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    @property
    def wall_clock_s(self) -> float:
        """Summed cell time. Not elapsed time: lanes and workers overlap."""
        return float(sum(c.wall_clock_s for c in self.cells))

    @property
    def config_hashes(self) -> dict[tuple[str, str, int, str], str]:
        """``(asset, model, horizon, arm) -> config_hash`` for every cell that has one."""
        return {
            (c.asset, c.model, c.horizon, c.arm): c.config_hash
            for c in self.cells
            if c.config_hash is not None
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_json() for c in self.cells])

    def as_json(self) -> dict[str, Any]:
        return {
            "n_cells": len(self.cells),
            "n_computed": self.n_computed,
            "n_cached": self.n_cached,
            "n_failed": self.n_failed,
            "cells": [c.as_json() for c in self.cells],
        }

    def write(self, path: str | Path) -> Path:
        """Write the manifest as JSON, through a temp file and ``os.replace``.

        Same write discipline as the store: a killed run leaves the old
        manifest or the new one, never half of either.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(self.as_json(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, destination)
        return destination

    def __str__(self) -> str:
        return (
            f"RunManifest({len(self.cells)} cells: {self.n_computed} computed, "
            f"{self.n_cached} cached, {self.n_failed} failed)"
        )


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _CellTask:
    """One cell plus everything running it needs. Picklable; ``eq=False``: pandas."""

    index: int
    cell: Cell
    data: AssetData
    store: ResultsStore | None
    overwrite: bool


def _run_cell(task: _CellTask) -> CellOutcome:
    """Score one cell, or record why it could not be scored.

    Module-level and taking a single dataclass, because the process backend
    pickles both the function and its argument (:mod:`volbench.execute`).

    Every cell gets a fresh :class:`SerialExecutor` for its refit blocks —
    execute.py's rule 1. This function may itself be running inside a worker,
    and a pool-backed executor nested inside a bounded pool deadlocks.
    """
    cell = task.cell
    started = time.perf_counter()
    try:
        fit_series = (
            task.data.fit_series(cell.arm.invalid_target_policy)
            if cell.model.fits_on_variance
            else None
        )
        frame = run_backtest(
            cell.model.factory,
            task.data.returns,
            task.data.proxy,
            cell.splitter(),
            cell.seed,
            asset=cell.asset,
            proxy_name=task.data.proxy_name,
            data_spec=task.data.data_spec,
            fit_series=fit_series,
            levels=cell.levels,
            executor=SerialExecutor(),
            store=task.store,
            overwrite=task.overwrite,
            recondition=cell.arm.recondition,
        )
    except Exception as exc:  # per-cell isolation — see the module docstring
        return _outcome(
            task,
            status="failed",
            config_hash=None,
            n_rows=0,
            n_missing=0,
            elapsed=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return _outcome(
        task,
        status="cached" if bool(frame.attrs.get("cached", False)) else "computed",
        config_hash=str(frame.attrs["config_hash"]),
        n_rows=len(frame),
        n_missing=_count_unscorable(frame),
        elapsed=time.perf_counter() - started,
        error=None,
    )


def _count_unscorable(frame: pd.DataFrame) -> int:
    """Rows whose ``missing_reason`` is non-empty.

    A *scored* row carries ``missing_reason == ""``, not NaN (``evaluate``
    pins the column to ``str``), so counting non-null values here would report
    every row as unscorable — which is exactly the kind of statistic that gets
    copied into a table and believed.
    """
    if "missing_reason" not in frame.columns:
        return 0
    return int((frame["missing_reason"].fillna("") != "").sum())


def _outcome(
    task: _CellTask,
    *,
    status: CellStatus,
    config_hash: str | None,
    n_rows: int,
    n_missing: int,
    elapsed: float,
    error: str | None,
) -> CellOutcome:
    cell = task.cell
    return CellOutcome(
        index=task.index,
        asset=cell.asset,
        model=cell.model.label,
        horizon=cell.horizon,
        arm=cell.arm.label,
        lane=cell.lane,
        status=status,
        config_hash=config_hash,
        n_rows=n_rows,
        n_missing=n_missing,
        wall_clock_s=elapsed,
        error=error,
    )


def run_grid(
    grid: GridSpec,
    source: DataSource | Mapping[str, AssetData],
    store: ResultsStore | None = None,
    *,
    cpu_executor: Executor | None = None,
    gpu_executor: Executor | None = None,
    overwrite: bool = False,
    manifest_path: str | Path | None = None,
    on_cell: Callable[[CellOutcome], None] | None = None,
) -> RunManifest:
    """Run every cell of ``grid``, resumably, and return the manifest.

    Parameters
    ----------
    grid:
        The cross to run. Expanded in the grid's total order (see
        :meth:`GridSpec.cells`).
    source:
        A :class:`DataSource`, or a plain mapping of asset id to
        :class:`AssetData` (wrapped in a :class:`MappingDataSource`). Loaded
        once per asset, in this process, before any cell runs: one read, one
        content digest, and every cell of that asset provably scored against
        the same bytes.
    store:
        Where fragments land, and what makes the run resumable. A cell whose
        ``config_hash`` is already stored is not recomputed. ``None`` runs
        everything in memory — useful for a smoke run, useless for a grid.
    cpu_executor:
        Backend for ``lane="cpu"`` cells. Defaults to
        :class:`~volbench.execute.SerialExecutor`, so the default run is fully
        serial and reproducible; pass
        :class:`~volbench.execute.ProcessExecutor` to fan out.
    gpu_executor:
        Backend for ``lane="gpu"`` cells. Defaults to a
        :class:`~volbench.execute.SerialExecutor` — one GPU cell at a time,
        which is the point of the lane. Pass ``ProcessExecutor(workers=1)`` to
        get the same serialization in a separate process (and, with it, a fresh
        CUDA context per cell).
    overwrite:
        Recompute and replace stored fragments instead of skipping them. Off by
        default: the append-only store is what makes an interrupted grid
        resumable, and overwriting is how that guarantee gets lost.
    manifest_path:
        If given, the manifest is written there as JSON.
    on_cell:
        Called with each :class:`CellOutcome` as it is collected, in lane
        order. A progress hook; it cannot change what runs.

    Returns
    -------
    A :class:`RunManifest` in the grid's total order.

    Notes
    -----
    Lanes run in :data:`LANE_ORDER` — CPU first, then GPU — for the fork-safety
    reason in the module docstring. Within a lane, order is the executor's
    business; the manifest is sorted back into grid order regardless.

    Data loading is *not* isolated per cell in one respect: if
    ``source.load(asset)`` raises, that is a failure of the source and it
    propagates, because a grid that silently records every cell of an asset as
    failed because a path was wrong is a grid that wasted its whole run. What
    the *cell* isolates is everything downstream of a successful load.
    """
    resolved: DataSource = (
        MappingDataSource(source) if isinstance(source, Mapping) else source
    )
    cells = grid.cells()
    data = {asset: resolved.load(asset) for asset in sorted(grid.assets)}
    tasks = [
        _CellTask(
            index=index,
            cell=cell,
            data=data[cell.asset],
            store=store,
            overwrite=overwrite,
        )
        for index, cell in enumerate(cells)
    ]

    backends: dict[Lane, Executor] = {
        "cpu": cpu_executor if cpu_executor is not None else SerialExecutor(),
        "gpu": gpu_executor if gpu_executor is not None else SerialExecutor(),
    }
    outcomes: list[CellOutcome] = []
    for lane in LANE_ORDER:
        lane_tasks = [task for task in tasks if task.cell.lane == lane]
        if not lane_tasks:
            continue
        logger.info(
            "runner: %s lane, %d cells, backend %r", lane, len(lane_tasks), backends[lane]
        )
        for outcome in backends[lane].map(_run_cell, lane_tasks):
            if outcome.status == "failed":
                logger.warning(
                    "runner: cell %s failed: %s",
                    cells[outcome.index],
                    outcome.error,
                )
            if on_cell is not None:
                on_cell(outcome)
            outcomes.append(outcome)

    manifest = RunManifest(cells=tuple(sorted(outcomes, key=lambda o: o.index)))
    if manifest_path is not None:
        manifest.write(manifest_path)
    return manifest


def read_grid_results(store: ResultsStore, manifest: RunManifest) -> pd.DataFrame:
    """Every scored row the manifest's cells produced, in config-hash order.

    A convenience over :meth:`ResultsStore.read`, and deliberately not a
    ``read_all``: it returns *this grid's* cells, so a store shared with an
    earlier study does not silently widen the table an analysis runs on.
    """
    hashes = sorted(
        {
            c.config_hash
            for c in manifest.cells
            if c.config_hash is not None and store.has(c.config_hash)
        }
    )
    if not hashes:
        return pd.DataFrame()
    return pd.concat([store.read(h) for h in hashes], ignore_index=True)
