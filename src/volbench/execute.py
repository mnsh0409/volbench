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

**Backends.** :class:`SerialExecutor` is the reference. :class:`ProcessExecutor`
(added on ``feat/p2-runner``, D-028) is the local multiprocessing backend; a
Slurm-array-backed implementation belongs here too when it is built.
Evaluation code must not grow an ``n_jobs`` argument.

Two rules, both now enforced rather than only documented:

1. *No nesting.* A pool-backed executor mapping over cells must hand each cell
   a :class:`SerialExecutor` for its blocks. Nesting two pool-backed executors
   deadlocks on a bounded worker pool.
   :meth:`ProcessExecutor.map` raises inside a worker rather than trying it.
2. *No order dependence.* Results are sorted by key before they are written
   (:func:`volbench.results.normalize_frame`), so ``map`` may return in any
   order it likes — but it must return *all* results, one per item, and the
   ``i``-th result must correspond to the ``i``-th item.

Anything mapped through an :class:`Executor` must be picklable (module-level
functions and dataclasses, not closures or lambdas), because the process and
Slurm backends cross a process boundary.

The numpy kernel family (D-026)
===============================
Byte-identity is a claim *within one numpy SIMD kernel family*: numpy's
AVX-512-only float64 ``log``/``exp`` kernels differ from the x86-v3 ones in the
last ulp for some inputs, which moves the content digest of a computed proxy
and every config hash built on it. The Makefile and CI pin the family with
``NPY_DISABLE_CPU_FEATURES``; a worker process that selected a different family
would silently produce different hashes for the same experiment — a cache
*miss* rather than a wrong answer, but a wasted grid and a broken identity
claim.

:class:`ProcessExecutor` closes that hole twice over:

- it propagates ``NPY_DISABLE_CPU_FEATURES`` — exactly the parent's value,
  including "unset", because inventing one the parent did not have would make
  the pool disagree with a serial run in the same shell — so a worker that
  imports numpy afresh reads what the parent read; and
- every worker checks its own enabled dispatch targets against the parent's
  (:func:`kernel_signature`) and refuses the task if they differ, so the
  failure is a loud error naming both signatures rather than a silently
  orphaned fragment.

Under a plain ``fork`` there is a third, stronger guarantee — the worker
inherits the parent's already-initialized numpy, i.e. the dispatch decision
itself rather than the environment that produced it — which is why ``fork`` is
still offered. It is not the default, for the reason below.

Why the default start method is ``forkserver``, not ``fork``
============================================================
Measured on this project, not assumed: a parent that has **trained a LightGBM
model** and then forks produces workers that **deadlock** on their first
LightGBM call. LightGBM is OpenMP-backed, `fork` copies the OpenMP runtime's
locks in whatever state the parent's threads left them, and the child's first
parallel region waits forever on a lock no thread will release. The same run
completes in 0.19 s with ``OMP_NUM_THREADS=1`` set before the parent starts,
and in 1.2 s under ``forkserver`` — which is the shape of an inherited-state
problem, not a load problem. It is not hypothetical for this codebase: a
script that runs one grid serially and the next through a pool, or a test
session that does both, hits it exactly.

``forkserver`` is still fork-based — workers are forked from a *clean* server
process that never trained anything — so the failure cannot occur whatever the
parent did, at a cost of about one second per pool (once per lane, against
grids measured in hours). ``fork`` remains available and is faster to start;
:meth:`ProcessExecutor.map` refuses it in a process that has already imported a
known fork-unsafe backend rather than letting it hang, which is precise here
because those backends are imported lazily inside ``fit`` (D-022) — so
"imported" means "already used". Workers under either method must be able to
import ``__main__`` safely; a script needs the usual
``if __name__ == "__main__":`` guard.

GPU contention is NOT this module's problem
===========================================
Nothing here knows what a cell computes, so nothing here can know that two
cells want the same GPU. Serializing the GPU-bound part of a grid is the
*runner's* job and is explicit in its configuration
(:class:`volbench.runner.ModelConfig.lane`, D-027) — never guessed from a
model's name here.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeVar, runtime_checkable

__all__ = [
    "DEFAULT_START_METHOD",
    "FORK_UNSAFE_MODULES",
    "Executor",
    "ProcessExecutor",
    "SerialExecutor",
    "kernel_signature",
]

T = TypeVar("T")
R = TypeVar("R")

#: The environment variable D-026 pins the numpy SIMD kernel family with.
KERNEL_PIN_VAR: Final = "NPY_DISABLE_CPU_FEATURES"

#: Backends whose native runtime makes a *plain* ``fork`` unsafe once this
#: process has used them (module docstring). Membership in ``sys.modules`` is a
#: precise test of "already used", not merely "available", because every
#: optional backend is imported inside ``fit`` and nowhere else (D-022).
FORK_UNSAFE_MODULES: Final = ("lightgbm",)

#: Start method used when none is given. See the module docstring for the
#: measurement behind it.
DEFAULT_START_METHOD: Final = "forkserver"


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
    is what the determinism canary in ``tests/test_evaluate.py`` pins down and
    what ``tests/test_runner.py::TestSerialParallelIdentity`` compares the
    process backend against, byte for byte.
    """

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        return [fn(item) for item in items]

    def __repr__(self) -> str:
        return "SerialExecutor()"


# --------------------------------------------------------------------------
# numpy kernel family (D-026)
# --------------------------------------------------------------------------


def kernel_signature() -> str:
    """Digest of the numpy SIMD dispatch targets *enabled in this process*.

    Two processes sharing this string compute ``log``/``exp`` with the same
    kernels and therefore agree bit for bit; two that do not can disagree in
    the last ulp, which moves every content digest downstream (D-026).

    ``numpy._core._multiarray_umath`` is private, so an unreadable or
    restructured build degrades to a signature over the pin variable itself
    rather than raising: this is a guard, and a guard that cannot run must not
    take the run down with it.
    """
    try:
        from numpy._core._multiarray_umath import (  # type: ignore[import-not-found]
            __cpu_baseline__,
            __cpu_dispatch__,
            __cpu_features__,
        )
    except Exception:  # pragma: no cover - numpy internals moved or unreadable
        return "unknown:" + os.environ.get(KERNEL_PIN_VAR, "")
    enabled = [target for target in sorted(__cpu_dispatch__) if __cpu_features__.get(target)]
    canonical = "baseline=" + ",".join(sorted(__cpu_baseline__)) + ";dispatch=" + ",".join(enabled)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]


def _kernel_env() -> dict[str, str | None]:
    """The parent's kernel pin, as something a worker can reapply verbatim.

    ``None`` means *unset*, and a worker sets it unset — inventing a value the
    parent did not have would make the pool disagree with a serial run in the
    same shell, which is precisely the failure this exists to prevent.
    """
    return {KERNEL_PIN_VAR: os.environ.get(KERNEL_PIN_VAR)}


# --------------------------------------------------------------------------
# process pool
# --------------------------------------------------------------------------

#: Set in a worker by :func:`_init_worker`; read by :meth:`ProcessExecutor.map`
#: to refuse rule 1 (no nesting) and by :func:`_apply` to check the kernel
#: family once per process rather than once per task.
_IN_WORKER = False
_CHECKED_KERNEL: str | None = None


def _init_worker(env: dict[str, str | None]) -> None:
    """Runs once per worker, before it touches numpy under a spawn start.

    Under ``fork`` numpy is already imported and its dispatch inherited, so
    this is a belt; under any other start method it is the brace.
    """
    global _IN_WORKER
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _IN_WORKER = True


def _require_importable_main(start_method: str) -> None:
    """Refuse a spawn-family pool where its workers could not start.

    Every start method other than ``fork`` re-imports the parent's ``__main__``
    in each worker, so a session whose ``__main__`` is not a file on disk — a
    REPL, ``python -``, a heredoc — cannot host one. Left alone that surfaces as
    ``BrokenProcessPool`` with a ``FileNotFoundError: '<stdin>'`` buried in the
    worker's traceback, which says nothing about what to do next.
    """
    main = sys.modules.get("__main__")
    if getattr(main, "__spec__", None) is not None:
        return
    path = getattr(main, "__file__", None)
    if path is not None and os.path.isfile(path):
        return
    raise RuntimeError(
        f"start_method={start_method!r} needs a worker to be able to import this process's "
        f"__main__, and it is {path!r} — a REPL, `python -`, or a heredoc. Run the grid from "
        "a script or module with the usual `if __name__ == \"__main__\":` guard, or pass "
        "start_method='fork' (faster, and safe as long as this process has not itself fitted "
        f"a model with an OpenMP backend: {list(FORK_UNSAFE_MODULES)})."
    )


@dataclass(frozen=True)
class _Payload:
    """One unit of work plus the parent's kernel signature to check against."""

    fn: Callable[[Any], Any]
    item: Any
    kernel: str | None


def _apply(payload: _Payload) -> Any:
    """Worker-side entry point: verify the kernel family once, then do the work."""
    global _CHECKED_KERNEL
    if payload.kernel is not None and payload.kernel != _CHECKED_KERNEL:
        local = kernel_signature()
        if local != payload.kernel:
            raise RuntimeError(
                "worker numpy SIMD kernel family differs from the parent's "
                f"({local} != {payload.kernel}). Results computed here would carry "
                f"different content digests and different config hashes (D-026). "
                f"{KERNEL_PIN_VAR}={os.environ.get(KERNEL_PIN_VAR)!r} in this worker."
            )
        _CHECKED_KERNEL = local
    return payload.fn(payload.item)


@dataclass(frozen=True)
class ProcessExecutor:
    """Local multiprocessing backend (D-028). Same results as
    :class:`SerialExecutor`, on more cores.

    Parameters
    ----------
    workers:
        Worker processes. ``None`` means :func:`os.cpu_count`. The pool is
        created per :meth:`map` call and torn down after it, so an executor
        object holds no OS resources between calls and is safe to keep in a
        config.
    start_method:
        ``"forkserver"`` by default: workers are forked from a clean server
        process, so nothing the parent did to a native threaded runtime can
        reach them. ``"fork"`` is faster to start and inherits the parent's
        initialized numpy (the strongest form of the D-026 guarantee), but it
        deadlocks in a process that has already trained a LightGBM model —
        measured, see the module docstring — so :meth:`map` refuses it there.
        ``"spawn"`` also works; every method other than ``fork`` relies on the
        environment propagation and the per-worker kernel check for D-026.
    check_kernel_family:
        Verify each worker's enabled numpy dispatch targets against the
        parent's before it runs anything (D-026). On by default; turning it off
        is only for measuring the difference on purpose.

    Not a GPU scheduler. Routing GPU-bound cells away from the fan-out is
    :mod:`volbench.runner`'s job, from explicit configuration (D-027).
    """

    workers: int | None = None
    start_method: str = DEFAULT_START_METHOD
    check_kernel_family: bool = True

    def __post_init__(self) -> None:
        if self.workers is not None and self.workers < 1:
            raise ValueError(f"workers must be >= 1 or None, got {self.workers}")
        available = multiprocessing.get_all_start_methods()
        if self.start_method not in available:
            raise ValueError(
                f"start method {self.start_method!r} is not available on this platform; "
                f"have {sorted(available)}"
            )

    @property
    def n_workers(self) -> int:
        """The worker count this executor will actually ask for."""
        return self.workers if self.workers is not None else (os.cpu_count() or 1)

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        if _IN_WORKER:
            raise RuntimeError(
                "ProcessExecutor.map was called inside a worker process. Pool-backed "
                "executors must not nest (volbench.execute rule 1): a cell running in a "
                "pool has to hand its refit blocks a SerialExecutor, or a bounded pool "
                "deadlocks waiting on itself."
            )
        if self.start_method != "fork":
            _require_importable_main(self.start_method)
        if self.start_method == "fork":
            used = [name for name in FORK_UNSAFE_MODULES if name in sys.modules]
            if used:
                raise RuntimeError(
                    f"start_method='fork' is unsafe in this process: {used} has already been "
                    "used here, and a forked worker deadlocks on its first call into an "
                    "OpenMP-backed backend whose locks were copied mid-flight (measured; see "
                    "volbench.execute's module docstring). Use the default "
                    f"start_method={DEFAULT_START_METHOD!r}, which forks workers from a clean "
                    "server process, or run the pool from a process that has not fitted a "
                    "model itself."
                )
        materialized = list(items)
        if not materialized:
            # No pool, no fork: mapping nothing must cost nothing, and an empty
            # grid is a perfectly ordinary resumed run.
            return []
        # Imported here rather than at module top so that `import volbench` on a
        # machine that never parallelizes anything pays nothing for it.
        from concurrent.futures import ProcessPoolExecutor

        kernel = kernel_signature() if self.check_kernel_family else None
        payloads = [_Payload(fn=fn, item=item, kernel=kernel) for item in materialized]
        context = multiprocessing.get_context(self.start_method)
        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(_kernel_env(),),
        ) as pool:
            # `.map` yields in item order and re-raises the first exception, so
            # the Executor contract (one result per item, in item order, faults
            # not swallowed) holds without any reordering here.
            return list(pool.map(_apply, payloads))

    def __repr__(self) -> str:
        return (
            f"ProcessExecutor(workers={self.workers!r}, start_method={self.start_method!r}, "
            f"check_kernel_family={self.check_kernel_family!r})"
        )
