"""Executor seam: the contract every Phase 2 backend must satisfy.

These tests pin the interface, not the implementation. When the
multiprocessing and Slurm-array backends land, they should be parametrized
into this file unchanged — anything they cannot pass is a backend that would
break the backend-invariance claim in D-011.
"""

from collections.abc import Iterator

import pytest

from volbench.execute import Executor, SerialExecutor


def double(x: int) -> int:
    """Module-level (i.e. picklable) — Phase 2 backends cross a process boundary."""
    return 2 * x


def boom(x: int) -> int:
    raise ValueError(f"boom {x}")


def test_serial_executor_satisfies_the_protocol() -> None:
    assert isinstance(SerialExecutor(), Executor)


def test_map_returns_one_result_per_item_in_item_order() -> None:
    items = [3, 1, 2, 5, 4]
    assert SerialExecutor().map(double, items) == [6, 2, 4, 10, 8]


def test_map_accepts_any_iterable_not_just_sequences() -> None:
    def gen() -> Iterator[int]:
        yield from (1, 2, 3)

    assert SerialExecutor().map(double, gen()) == [2, 4, 6]


def test_map_of_nothing_is_empty_not_an_error() -> None:
    assert SerialExecutor().map(double, []) == []


def test_map_propagates_exceptions() -> None:
    """A failed cell must surface, not be swallowed into a partial result set."""
    with pytest.raises(ValueError, match="boom 2"):
        SerialExecutor().map(boom, [2])


def test_repr_is_stable() -> None:
    assert repr(SerialExecutor()) == "SerialExecutor()"
