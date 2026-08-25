"""Executor seam: the contract every backend must satisfy.

The contract tests are parametrized over every backend, which is the point:
they pin the interface, not an implementation, and a Slurm-array backend
should join the list unchanged. Anything a backend cannot pass is a backend
that would break the backend-invariance claim in D-011.

``TestProcessExecutor`` adds what is specific to the local multiprocessing
backend (D-028): the no-nesting rule, the numpy kernel-family propagation
D-026 requires, and worker-count handling. The *identity* gate — parallel
fragments byte-identical to serial ones over a real grid — lives in
``tests/test_runner.py::TestSerialParallelIdentity``, because it needs a grid
to be a claim about anything.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from volbench.execute import (
    DEFAULT_START_METHOD,
    KERNEL_PIN_VAR,
    Executor,
    ProcessExecutor,
    SerialExecutor,
    kernel_signature,
)

BACKENDS = [
    pytest.param(SerialExecutor(), id="serial"),
    pytest.param(ProcessExecutor(workers=2), id="process"),
]


def double(x: int) -> int:
    """Module-level (i.e. picklable) — the process backend crosses a boundary."""
    return 2 * x


def boom(x: int) -> int:
    raise ValueError(f"boom {x}")


def worker_pid(_: int) -> int:
    return os.getpid()


def read_kernel_pin(_: int) -> tuple[str | None, str]:
    """What the worker sees for the D-026 pin, and which kernels it enabled."""
    return os.environ.get(KERNEL_PIN_VAR), kernel_signature()


def nested_map(_: int) -> list[int]:
    """Rule 1: a pool-backed executor inside a worker must refuse, not deadlock."""
    return ProcessExecutor(workers=2).map(double, [1, 2])


# --------------------------------------------------------------------------
# the contract — every backend
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
class TestExecutorContract:
    def test_satisfies_the_protocol(self, backend: Executor) -> None:
        assert isinstance(backend, Executor)

    def test_map_returns_one_result_per_item_in_item_order(self, backend: Executor) -> None:
        items = [3, 1, 2, 5, 4]
        assert backend.map(double, items) == [6, 2, 4, 10, 8]

    def test_map_accepts_any_iterable_not_just_sequences(self, backend: Executor) -> None:
        def gen() -> Iterator[int]:
            yield from (1, 2, 3)

        assert backend.map(double, gen()) == [2, 4, 6]

    def test_map_of_nothing_is_empty_not_an_error(self, backend: Executor) -> None:
        assert backend.map(double, []) == []

    def test_map_propagates_exceptions(self, backend: Executor) -> None:
        """A failed cell must surface, not be swallowed into a partial result set."""
        with pytest.raises(ValueError, match="boom 2"):
            backend.map(boom, [2])

    def test_the_same_items_twice_give_the_same_results(self, backend: Executor) -> None:
        items = list(range(20))
        assert backend.map(double, items) == backend.map(double, items)


class TestSerialExecutor:
    def test_repr_is_stable(self) -> None:
        assert repr(SerialExecutor()) == "SerialExecutor()"

    def test_it_runs_in_this_process(self) -> None:
        assert SerialExecutor().map(worker_pid, [0]) == [os.getpid()]


# --------------------------------------------------------------------------
# the local multiprocessing backend (D-028)
# --------------------------------------------------------------------------


class TestProcessExecutor:
    def test_work_really_leaves_this_process(self) -> None:
        pids = ProcessExecutor(workers=2).map(worker_pid, list(range(8)))
        assert os.getpid() not in pids

    def test_worker_count_is_configurable_and_bounded(self) -> None:
        assert len(set(ProcessExecutor(workers=1).map(worker_pid, list(range(6))))) == 1
        assert len(set(ProcessExecutor(workers=2).map(worker_pid, list(range(24))))) <= 2

    def test_worker_count_defaults_to_the_machines(self) -> None:
        assert ProcessExecutor().n_workers == (os.cpu_count() or 1)
        assert ProcessExecutor(workers=3).n_workers == 3

    def test_a_nonsense_worker_count_or_start_method_is_refused(self) -> None:
        with pytest.raises(ValueError, match="workers"):
            ProcessExecutor(workers=0)
        with pytest.raises(ValueError, match="start method"):
            ProcessExecutor(start_method="teleport")

    def test_the_default_start_method_is_forkserver(self) -> None:
        """Measured, not preferred: a parent that has trained a LightGBM model
        and then plain-forks produces workers that deadlock on their first
        LightGBM call (OpenMP locks copied mid-flight). ``forkserver`` forks
        workers from a clean server process, so nothing the parent did to a
        native runtime can reach them."""
        assert ProcessExecutor().start_method == DEFAULT_START_METHOD == "forkserver"
        assert ProcessExecutor(start_method="fork").start_method == "fork"

    def test_an_empty_map_starts_no_processes(self) -> None:
        """A fully-resumed grid maps nothing; that must cost nothing."""
        assert ProcessExecutor(workers=64).map(double, []) == []

    def test_repr_is_stable(self) -> None:
        assert repr(ProcessExecutor(workers=2)) == (
            "ProcessExecutor(workers=2, start_method='forkserver', check_kernel_family=True)"
        )


class TestForkIsRefusedWhereItWouldHang:
    """The measured hazard, turned into a message instead of a deadlock.

    A parent that has trained a LightGBM model leaves an OpenMP runtime whose
    locks a plain ``fork`` copies mid-flight; the child's first parallel region
    then waits forever. There is deliberately no test that *reproduces* the
    deadlock — a hanging test is worse than the bug — so what is pinned is the
    guard, and the fact that the detector is exact: every optional backend is
    imported inside ``fit`` (D-022), so "in ``sys.modules``" means "already
    used in this process".
    """

    def test_fork_is_refused_once_an_openmp_backend_has_been_used(self) -> None:
        pytest.importorskip("lightgbm")
        import lightgbm  # noqa: F401 - the import is the precondition

        with pytest.raises(RuntimeError, match="unsafe in this process"):
            ProcessExecutor(workers=2, start_method="fork").map(double, [1, 2])

    def test_the_message_names_the_way_out(self) -> None:
        pytest.importorskip("lightgbm")
        import lightgbm  # noqa: F401

        with pytest.raises(RuntimeError, match="forkserver"):
            ProcessExecutor(workers=2, start_method="fork").map(double, [1])

    def test_the_default_is_not_refused(self) -> None:
        pytest.importorskip("lightgbm")
        import lightgbm  # noqa: F401

        assert ProcessExecutor(workers=2).map(double, [1, 2]) == [2, 4]


class TestWorkersMustBeAbleToImportMain:
    """Every start method other than ``fork`` re-imports the parent's
    ``__main__`` in each worker, so a REPL, ``python -`` or a heredoc cannot
    host a pool. Left alone that arrives as ``BrokenProcessPool`` with a
    ``FileNotFoundError: '<stdin>'`` buried in a worker traceback, which says
    nothing about what to do; the guard says it up front."""

    def test_a_session_without_an_importable_main_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import ModuleType

        from volbench.execute import _require_importable_main

        stdin_main = ModuleType("__main__")
        stdin_main.__file__ = "<stdin>"
        monkeypatch.setitem(sys.modules, "__main__", stdin_main)

        with pytest.raises(RuntimeError, match="import this process's __main__"):
            _require_importable_main("forkserver")
        # The message has to name both ways out, or it is only half a diagnosis.
        with pytest.raises(RuntimeError, match="if __name__"):
            _require_importable_main("spawn")
        with pytest.raises(RuntimeError, match="start_method='fork'"):
            _require_importable_main("spawn")

    def test_fork_needs_no_importable_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """It inherits the interpreter rather than rebuilding it, which is the
        one thing plain fork is unambiguously better at."""
        from types import ModuleType

        stdin_main = ModuleType("__main__")
        stdin_main.__file__ = "<stdin>"
        monkeypatch.setitem(sys.modules, "__main__", stdin_main)
        monkeypatch.delitem(sys.modules, "lightgbm", raising=False)

        assert ProcessExecutor(workers=2, start_method="fork").map(double, [5]) == [10]

    def test_a_real_script_or_module_passes(self) -> None:
        from volbench.execute import _require_importable_main

        _require_importable_main("forkserver")  # pytest's own __main__ is a file


class TestKernelFamily:
    """D-026: digests are numpy-kernel-family dependent, so a worker that
    selected a different family would silently produce different config
    hashes for the same experiment."""

    def test_workers_see_the_parents_kernel_pin_and_the_same_kernels(self) -> None:
        seen = ProcessExecutor(workers=2).map(read_kernel_pin, list(range(6)))
        parent = (os.environ.get(KERNEL_PIN_VAR), kernel_signature())
        assert set(seen) == {parent}

    def test_the_pin_is_propagated_verbatim_including_when_it_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(KERNEL_PIN_VAR, "X86_V4 AVX512_ICL AVX512_SPR")
        pool = ProcessExecutor(workers=2, check_kernel_family=False)
        values = {v for v, _ in pool.map(read_kernel_pin, [0, 1])}
        assert values == {"X86_V4 AVX512_ICL AVX512_SPR"}

    def test_an_unset_pin_stays_unset_in_the_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inventing a value the parent did not have would make the pool
        disagree with a serial run in the same shell — the exact failure this
        propagation exists to prevent."""
        monkeypatch.delenv(KERNEL_PIN_VAR, raising=False)
        pool = ProcessExecutor(workers=2, check_kernel_family=False)
        values = {v for v, _ in pool.map(read_kernel_pin, [0, 1])}
        assert values == {None}

    def test_a_worker_on_a_different_kernel_family_refuses_the_work(self) -> None:
        """What the check is for, exercised directly on the worker-side entry
        point rather than by trying to engineer a real mismatch: numpy reads
        the pin once at import, and a pool's server process caches its own
        import, so a mid-process env change is not a reliable way to produce
        one. The failure that matters is a worker whose kernels differ from the
        parent's — it must refuse rather than write a fragment whose content
        digest nobody can reproduce (D-026)."""
        from volbench.execute import _apply, _Payload

        honest = _Payload(fn=double, item=21, kernel=kernel_signature())
        assert _apply(honest) == 42

        impostor = _Payload(fn=double, item=21, kernel="0123456789abcdef")
        with pytest.raises(RuntimeError, match="kernel family"):
            _apply(impostor)

    def test_the_check_can_be_turned_off_for_a_deliberate_measurement(self) -> None:
        from volbench.execute import _apply, _Payload

        assert _apply(_Payload(fn=double, item=3, kernel=None)) == 6
        assert ProcessExecutor(workers=2, check_kernel_family=False).map(double, [4]) == [8]


class TestNoNesting:
    """Rule 1 of the module: pool-backed executors must not nest.

    Enforced rather than documented, because the failure it prevents is a
    deadlock — a bounded pool waiting on itself — which is indistinguishable
    from a slow grid until someone kills it.
    """

    def test_a_pool_inside_a_worker_raises_rather_than_deadlocking(self) -> None:
        with pytest.raises(RuntimeError, match="must not nest"):
            ProcessExecutor(workers=2).map(nested_map, [0])

    def test_nesting_is_fine_in_the_parent(self) -> None:
        """The guard is about *worker* processes, not about using the backend
        twice: a runner maps its CPU lane and then its GPU lane."""
        pool: Any = ProcessExecutor(workers=2)
        assert pool.map(double, [1]) == [2]
        assert pool.map(double, [2]) == [4]
