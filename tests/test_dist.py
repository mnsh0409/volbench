"""Distribution object: correctness of scores against brute force and closed forms."""

import itertools
import math

import numpy as np
import pytest

from volbench import Distribution, Empirical, Normal


def brute_force_crps(samples: np.ndarray, y: float) -> float:
    term1 = np.mean(np.abs(samples - y))
    term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
    return float(term1 - term2)


class TestNormal:
    def test_crps_closed_form_matches_sampling(self) -> None:
        d = Distribution.from_normal(mu=1.5, sigma=2.0)
        y = 0.7
        approx = Empirical(samples=np.sort(d.sample(200_000, seed=1))).crps(y)
        assert d.crps(y) == pytest.approx(approx, rel=5e-3)

    def test_crps_known_value_standard_normal_at_zero(self) -> None:
        # CRPS(N(0,1), 0) = 2*phi(0) - 1/sqrt(pi) = sqrt(2/pi) - 1/sqrt(pi)
        d = Normal(mu=0.0, sigma=1.0)
        expected = math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi)
        assert d.crps(0.0) == pytest.approx(expected, abs=1e-12)

    def test_crps_positive_and_min_near_mean(self) -> None:
        d = Normal(mu=0.0, sigma=1.0)
        assert d.crps(0.0) > 0.0
        assert d.crps(0.0) < d.crps(3.0) < d.crps(6.0)

    def test_log_score_matches_density(self) -> None:
        d = Normal(mu=0.3, sigma=1.7)
        y = -0.9
        density = math.exp(-0.5 * ((y - d.mu) / d.sigma) ** 2) / (
            d.sigma * math.sqrt(2 * math.pi)
        )
        assert d.log_score(y) == pytest.approx(-math.log(density), abs=1e-12)

    def test_quantile_cdf_roundtrip(self) -> None:
        d = Normal(mu=-2.0, sigma=0.5)
        for tau in (0.01, 0.025, 0.05, 0.5, 0.95):
            assert d.cdf(d.quantile(tau)) == pytest.approx(tau, abs=1e-9)

    def test_invalid_params_rejected(self) -> None:
        with pytest.raises(ValueError):
            Normal(mu=0.0, sigma=0.0)
        with pytest.raises(ValueError):
            Normal(mu=float("nan"), sigma=1.0)


class TestEmpirical:
    def test_crps_matches_brute_force(self) -> None:
        rng = np.random.default_rng(7)
        samples = rng.standard_t(df=4, size=501)  # odd size, fat tails
        d = Distribution.from_samples(samples)
        for y in (-3.0, 0.0, 1.234):
            assert d.crps(y) == pytest.approx(brute_force_crps(np.sort(samples), y), rel=1e-10)

    def test_crps_degenerate_ensemble_is_absolute_error(self) -> None:
        d = Distribution.from_samples([2.0, 2.0, 2.0, 2.0])
        assert d.crps(5.0) == pytest.approx(3.0)

    def test_cdf_monotone(self) -> None:
        d = Distribution.from_samples(np.random.default_rng(0).normal(size=100))
        xs = np.linspace(-4, 4, 50)
        cdfs = [d.cdf(float(x)) for x in xs]
        assert all(a <= b for a, b in itertools.pairwise(cdfs))

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError):
            Distribution.from_samples([1.0])
        with pytest.raises(ValueError):
            Distribution.from_samples([1.0, float("inf")])

    def test_log_score_undefined(self) -> None:
        d = Distribution.from_samples([0.0, 1.0])
        with pytest.raises(NotImplementedError):
            d.log_score(0.5)


class TestQuantileGrid:
    def test_recovers_normal_crps_on_dense_grid(self) -> None:
        ref = Normal(mu=0.0, sigma=1.0)
        taus = np.linspace(0.001, 0.999, 2001)
        values = np.array([ref.quantile(float(t)) for t in taus])
        d = Distribution.from_quantiles(taus, values)
        for y in (-1.0, 0.0, 2.0):
            assert d.crps(y) == pytest.approx(ref.crps(y), rel=2e-2)

    def test_quantile_interpolation_and_validation(self) -> None:
        d = Distribution.from_quantiles([0.25, 0.75], [1.0, 3.0])
        assert d.quantile(0.5) == pytest.approx(2.0)
        with pytest.raises(ValueError):
            Distribution.from_quantiles([0.25, 0.5], [1.0, 0.0])  # decreasing quantile values
        with pytest.raises(ValueError):
            Distribution.from_quantiles([0.0, 0.5], [0.0, 1.0])  # tau at boundary

    def test_pinball_via_interface(self) -> None:
        d = Distribution.from_quantiles([0.05, 0.5, 0.95], [-1.6, 0.0, 1.6])
        # y above the 5% quantile: loss = tau * (y - q)
        assert d.pinball(y=0.0, tau=0.05) == pytest.approx(0.05 * 1.6)


class TestSamplingDeterminism:
    def test_same_seed_same_draws(self) -> None:
        d = Normal(mu=0.0, sigma=1.0)
        a = d.sample(1000, seed=42)
        b = d.sample(1000, seed=42)
        assert np.array_equal(a, b)
        c = d.sample(1000, seed=43)
        assert not np.array_equal(a, c)


class TestArrayFieldHashEq:
    """Regression: Empirical/QuantileGrid carry numpy array fields, so
    dataclass's default generated __eq__/__hash__ (field tuples compared
    element-wise) raise on any array with more than one element. Both are
    declared eq=False and fall back to identity semantics instead of
    crashing (see the matching Origin test in test_splitter.py).
    """

    def test_empirical_hash_and_eq_do_not_raise(self) -> None:
        e1 = Distribution.from_samples([1.0, 2.0, 3.0])
        e2 = Distribution.from_samples([4.0, 5.0, 6.0])
        assert hash(e1) == hash(e1)
        assert e1 == e1
        assert e1 != e2
        assert e1 in {e1}

    def test_quantile_grid_hash_and_eq_do_not_raise(self) -> None:
        q1 = Distribution.from_quantiles([0.25, 0.75], [1.0, 3.0])
        q2 = Distribution.from_quantiles([0.25, 0.75], [2.0, 4.0])
        assert hash(q1) == hash(q1)
        assert q1 == q1
        assert q1 != q2
        assert q1 in {q1}
