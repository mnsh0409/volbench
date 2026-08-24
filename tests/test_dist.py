"""Distribution object: correctness of scores against brute force and closed forms."""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate, special

from volbench import Distribution, Empirical, Normal, StudentT


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


class TestStudentT:
    """Same rigor as ``Normal``: closed forms checked against brute force and
    against independent representations, never against themselves."""

    @pytest.mark.parametrize("nu", [1.5, 2.5, 3.0, 5.0, 8.0, 20.0, 100.0])
    @pytest.mark.parametrize("y", [-2.0, 0.3, 4.0])
    def test_crps_closed_form_matches_numerical_integration(self, nu: float, y: float) -> None:
        """Two independent witnesses for the closed form:

        - the quantile representation ``2 * int_0^1 pinball_tau(y - Q(tau)) dtau``
          (Laio & Tamea, 2007), integrated with the kink at ``tau = F(y)`` as a
          breakpoint;
        - the CDF representation ``int (F(x) - 1{x >= y})^2 dx``.

        Both are adaptive quadratures of *different* functions of the same
        law, so agreement to 1e-7 is not the formula checking itself.
        """
        d = StudentT(loc=0.3, scale=1.7, df=nu)
        closed = d.crps(y)
        via_quantiles, _ = integrate.quad(
            lambda t: d.pinball(y, t), 0.0, 1.0, points=[d.cdf(y)], limit=500, epsabs=1e-13
        )
        left, _ = integrate.quad(lambda x: d.cdf(x) ** 2, -np.inf, y, limit=500, epsabs=1e-13)
        right, _ = integrate.quad(
            lambda x: (1.0 - d.cdf(x)) ** 2, y, np.inf, limit=500, epsabs=1e-13
        )
        assert closed == pytest.approx(2.0 * via_quantiles, rel=1e-7)
        assert closed == pytest.approx(left + right, rel=1e-7)

    def test_crps_tends_to_the_gaussian_closed_form(self) -> None:
        """The beta-function constant must go to 1/sqrt(pi) — the case that a
        naive ``B(...)`` would have overflowed."""
        for y in (-2.0, 0.3, 4.0):
            t = StudentT(loc=0.3, scale=1.7, df=1e7).crps(y)
            n = Normal(mu=0.3, sigma=1.7).crps(y)
            assert t == pytest.approx(n, rel=1e-6)

    def test_crps_closed_form_matches_sampling(self) -> None:
        d = StudentT(loc=0.0, scale=1.0, df=5.0)
        approx = Empirical(samples=np.sort(d.sample(400_000, seed=1))).crps(0.7)
        assert d.crps(0.7) == pytest.approx(approx, rel=1e-2)  # fat tails: slow MC convergence

    def test_same_variance_t_is_more_peaked_not_more_forgiving_under_crps(self) -> None:
        """CRPS is an L1 score, and this is the opposite of the folk intuition.

        A t(3) with the *same variance* as N(0, 1) has scale 0.577: its
        variance comes from rare huge draws, so its typical spread
        ``E|X - X'|`` is *smaller* than the normal's. It therefore scores
        better at the center and worse on a far outlier — for large ``|y|``,
        ``CRPS(y) -> |y| - E|X - X'| / 2``, and the t's spread term (0.478)
        sits below the normal's ``1/sqrt(pi)`` (0.564). Pinned so nobody
        "corrects" the closed form toward the intuition later.
        """
        t = StudentT.from_variance(0.0, 1.0, df=3.0)
        n = Normal(mu=0.0, sigma=1.0)
        assert t.scale == pytest.approx(math.sqrt(1.0 / 3.0))
        assert t.crps(0.0) < n.crps(0.0)
        for y in (3.0, 6.0, 10.0):
            assert t.crps(y) > n.crps(y)
        spread_t = 50.0 - t.crps(50.0)
        spread_n = 50.0 - n.crps(50.0)
        assert spread_n == pytest.approx(1.0 / math.sqrt(math.pi), abs=1e-6)
        assert 0.0 < spread_t < spread_n

    def test_log_score_matches_density(self) -> None:
        d = StudentT(loc=0.3, scale=1.7, df=4.5)
        y = -0.9
        z = (y - d.loc) / d.scale
        log_density = (
            float(special.gammaln((d.df + 1.0) / 2.0))
            - float(special.gammaln(d.df / 2.0))
            - 0.5 * math.log(d.df * math.pi)
            - math.log(d.scale)
            - (d.df + 1.0) / 2.0 * math.log1p(z * z / d.df)
        )
        assert d.log_score(y) == pytest.approx(-log_density, abs=1e-12)

    def test_quantile_cdf_roundtrip(self) -> None:
        d = StudentT(loc=-2.0, scale=0.5, df=3.0)
        for tau in (0.01, 0.025, 0.05, 0.5, 0.95):
            assert d.cdf(d.quantile(tau)) == pytest.approx(tau, abs=1e-9)
        assert d.quantile(0.5) == pytest.approx(d.loc)

    def test_moments_are_analytic(self) -> None:
        d = StudentT(loc=0.3, scale=1.7, df=5.0)
        assert d.mean() == 0.3
        assert d.variance() == pytest.approx(1.7**2 * 5.0 / 3.0, rel=1e-15)
        # Sampling agrees, so the closed forms describe the law sample() draws from.
        draws = d.sample(400_000, seed=3)
        assert float(np.mean(draws)) == pytest.approx(d.mean(), abs=0.02)
        assert float(np.var(draws)) == pytest.approx(d.variance(), rel=0.05)

    def test_from_variance_round_trips_exactly_through_variance(self) -> None:
        for df in (2.02, 3.0, 5.0, 8.0, 20.0):
            d = StudentT.from_variance(0.0, 1e-4, df)
            assert d.variance() == pytest.approx(1e-4, rel=1e-12)
            assert d.df == df and d.loc == 0.0
            assert d.scale < math.sqrt(1e-4)  # scale is not the standard deviation

    def test_variance_is_a_clear_error_where_it_is_undefined(self) -> None:
        for df in (1.5, 2.0):
            d = StudentT(loc=0.0, scale=1.0, df=df)
            assert d.mean() == 0.0  # the mean exists for df > 1
            assert math.isfinite(d.crps(0.5))  # and so does the CRPS
            with pytest.raises(ValueError, match="undefined for df <= 2"):
                d.variance()
            with pytest.raises(ValueError, match="needs df > 2"):
                StudentT.from_variance(0.0, 1.0, df)

    def test_invalid_params_rejected(self) -> None:
        with pytest.raises(ValueError, match="df > 1"):
            StudentT(loc=0.0, scale=1.0, df=1.0)
        with pytest.raises(ValueError, match="df > 1"):
            StudentT(loc=0.0, scale=1.0, df=0.5)
        with pytest.raises(ValueError, match="scale > 0"):
            StudentT(loc=0.0, scale=0.0, df=5.0)
        with pytest.raises(ValueError, match="finite"):
            StudentT(loc=float("nan"), scale=1.0, df=5.0)
        with pytest.raises(ValueError, match="finite"):
            StudentT(loc=0.0, scale=1.0, df=float("inf"))
        with pytest.raises(ValueError, match="variance > 0"):
            StudentT.from_variance(0.0, -1.0, 5.0)
        with pytest.raises(ValueError):
            StudentT(loc=0.0, scale=1.0, df=5.0).quantile(1.0)

    def test_constructor_on_the_base_class(self) -> None:
        d = Distribution.from_student_t(0.3, 1.7, 5)
        assert isinstance(d, StudentT)
        assert d.df == 5.0

    def test_value_equality_and_hash(self) -> None:
        assert StudentT(0.0, 1.0, 5.0) == StudentT(0.0, 1.0, 5.0)
        assert hash(StudentT(0.0, 1.0, 5.0)) == hash(StudentT(0.0, 1.0, 5.0))
        assert StudentT(0.0, 1.0, 5.0) != StudentT(0.0, 1.0, 6.0)

    def test_pinball_via_interface(self) -> None:
        d = StudentT(loc=0.0, scale=1.0, df=5.0)
        q = d.quantile(0.05)
        assert d.pinball(q - 1.0, 0.05) == pytest.approx(0.95)
        assert d.pinball(q + 1.0, 0.05) == pytest.approx(0.05)


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

    def test_student_t_same_seed_same_draws(self) -> None:
        d = StudentT(loc=0.0, scale=1.0, df=4.0)
        assert np.array_equal(d.sample(1000, seed=42), d.sample(1000, seed=42))
        assert not np.array_equal(d.sample(1000, seed=42), d.sample(1000, seed=43))


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
