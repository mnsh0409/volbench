"""Execution seam — *where* work runs, never *what* work is done.

volbench's grid is embarrassingly parallel, but the parallelism must never be
allowed to leak into the evaluation logic: a backend that changed a number
would destroy the claim that results are backend-invariant (D-011). This
module is therefore the single, deliberately tiny interface between the two.

**Cells.** The unit of distributable work is a *cell*: one
``(asset, model, splitter, seed)`` combination. A cell is scored
independently of every other cell, and cells merge in the
:class:`~volbench.results.ResultsStore` purely by ``config_hash`` — no
cross-cell communication, no ordering requirement, no shared mutable state.
That is what lets the same cells run single-core, across local processes, or
as a Slurm array and produce byte-identical parquet (D-011 §PAPER
OPPORTUNITY).

**Sub-cell work.** Within one cell, :func:`volbench.evaluate.run_backtest`
splits its origins into *refit blocks* — a refit origin plus the origins that
reuse its fit — and maps those through an :class:`Executor` too. Blocks are
independent by construction (each block fits from its own training window,
supplied by ``RollingOriginSplitter``), so this decomposition is exactly
equivalent to a sequential loop and changes neither the number of fits nor any
score.

**Phase 2 will add** ``ProcessPoolExecutor``- and Slurm-array-backed
implementations of :class:`Executor`. They belong here and nowhere else;
evaluation code must not grow a ``n_jobs`` argument.

Two rules for the Phase 2 backends:

1. *No nesting.* A pool-backed executor mapping over cells must hand each cell
   a :class:`SerialExecutor` for its blocks. Nesting two pool-backed executors
   deadlocks on a bounded worker pool.
2. *No order dependence.* Results are sorted by key before they are written
   (:func:`volbench.results.normalize_frame`), so ``map`` may return in any
   order it likes — but it must return *all* results, one per item.

Anything mapped through an :class:`Executor` must be picklable (module-level
functions and dataclasses, not closures or lambdas), because the Phase 2
backends will cross a process boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["Executor", "SerialExecutor"]

T = TypeVar("T")
R = TypeVar("R")


@runtime_checkable
class Executor(Protocol):
    """Applies ``fn`` to every item, returning one result per item.

    Implementations may evaluate items in any order and on any machine. They
    must not reorder, drop, or deduplicate *results*: ``map`` returns exactly
    ``len(items)`` results, and result ``i`` corresponds to item ``i``.
    """

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Apply ``fn`` to each item and return the results in item order."""
        ...


class SerialExecutor:
    """The reference backend: a plain in-process loop.

    Every other backend is required to reproduce its output exactly, so this
    is what the determinism canary in ``tests/test_evaluate.py`` pins down.
    """

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        return [fn(item) for item in items]

    def __repr__(self) -> str:
        return "SerialExecutor()"
