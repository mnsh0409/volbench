"""RollingOriginSplitter: structural temporal-integrity guarantees.

These tests ARE the leakage contract. If any of them needs weakening to make
a feature work, the feature is wrong (see CLAUDE.md rule 1).
"""

import itertools

import numpy as np
import pytest

from volbench import RollingOriginSplitter


@pytest.mark.parametrize("window", [2, 5, 50])
@pytest.mark.parametrize("horizon", [1, 5])
@pytest.mark.parametrize("step", [1, 3])
@pytest.mark.parametrize("refit_every", [1, 4])
def test_structural_invariants(window: int, horizon: int, step: int, refit_every: int) -> None:
    n = 200
    sp = RollingOriginSplitter(
        window=window, horizon=horizon, step=step, refit_every=refit_every
    )
    origins = list(sp.split(n))
    assert len(origins) == sp.n_splits(n) > 0

    for k, o in enumerate(origins):
        # The leakage guarantee: everything trainable is <= origin < everything tested.
        assert o.train.max() == o.origin
        assert o.test.min() == o.origin + 1
        assert o.train.max() < o.test.min()
        # Exact shapes and bounds.
        assert o.train.size == window
        assert o.test.size == horizon
        assert o.train.min() == o.origin - window + 1
        assert o.train.min() >= 0
        assert o.test.max() <= n - 1
        # Contiguity.
        assert np.array_equal(o.train, np.arange(o.train.min(), o.origin + 1))
        assert np.array_equal(o.test, np.arange(o.origin + 1, o.origin + 1 + horizon))
        # Refit cadence: first origin refits, then every refit_every-th.
        assert o.refit == (k % refit_every == 0)

    # Origins advance by exactly `step`.
    starts = [o.origin for o in origins]
    assert all(b - a == step for a, b in itertools.pairwise(starts))
    # Last origin leaves a full horizon inside the series.
    assert starts[-1] <= n - 1 - horizon


def test_covers_forecastable_range_with_step_one() -> None:
    n, window, horizon = 50, 10, 1
    sp = RollingOriginSplitter(window=window, horizon=horizon)
    targets = sorted({int(t) for o in sp.split(n) for t in o.test})
    # Every index from the first forecastable target to the end is hit exactly once.
    assert targets == list(range(window, n))


def test_short_series_raises() -> None:
    sp = RollingOriginSplitter(window=10, horizon=5)
    with pytest.raises(ValueError, match="too short"):
        list(sp.split(14))
    # Exactly one origin fits at n = window + horizon.
    assert len(list(sp.split(15))) == 1


def test_invalid_params_rejected() -> None:
    for kwargs in (
        {"window": 1},
        {"window": 5, "horizon": 0},
        {"window": 5, "step": 0},
        {"window": 5, "refit_every": 0},
    ):
        with pytest.raises(ValueError):
            RollingOriginSplitter(**kwargs)


def test_future_corruption_canary() -> None:
    """Corrupting data strictly after origin T must not change what train exposes.

    This is the module-level version of the repo's leakage canary: the split
    indices for past origins are independent of future values by construction,
    so a pipeline that only consumes Origin.train cannot see corrupted data.
    """
    n = 120
    rng = np.random.default_rng(0)
    series = rng.normal(size=n)
    sp = RollingOriginSplitter(window=30, horizon=1)

    origins = [o for o in sp.split(n) if o.origin <= 80]
    views_clean = [series[o.train].copy() for o in origins]

    corrupted = series.copy()
    corrupted[81:] = 1e9  # poison the future
    views_poisoned = [corrupted[o.train] for o in origins]

    for clean, poisoned in zip(views_clean, views_poisoned, strict=True):
        assert np.array_equal(clean, poisoned)


def test_origin_hash_and_eq_do_not_raise() -> None:
    """Regression: Origin carries numpy array fields, so dataclass's default
    generated __eq__/__hash__ (which compare field tuples element-wise) raise
    on any array with more than one element. Origin is declared eq=False and
    falls back to identity semantics instead of crashing.
    """
    sp = RollingOriginSplitter(window=5, horizon=1)
    o1, o2 = list(sp.split(20))[:2]
    assert hash(o1) == hash(o1)
    assert o1 == o1
    assert o1 != o2
    assert o1 in {o1}
