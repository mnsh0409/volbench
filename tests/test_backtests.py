"""VaR/ES backtests: Kupiec POF, Christoffersen IND/CC, the FZ0 loss, ES.

Formula pins are hand transcriptions of the cited papers, written without
scipy's xlogy so an error in the convention handling shows up:

- Kupiec (1995): LR_POF against an explicit binomial log-likelihood.
- Christoffersen (1998): transition counts from an explicit loop, LRs from
  the explicit Markov likelihood, and the exact identity LR_cc = LR_uc + LR_ind.
- Patton, Ziegel & Chen (2019, eq. 6): the FZ0 value at the paper's own
  Figure 1 configuration, the ``L(kY, kv, ke) = L + log k`` identity from
  their remark after eq. (42), and the Figure 2 minimum at the true (VaR, ES)
  of a standard normal.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy import integrate, stats

from volbench.backtests import (
    MIN_EXPECTED_HITS,
    ChristoffersenResult,
    KupiecResult,
    SmallSampleWarning,
    VaRBacktest,
    christoffersen,
    expected_shortfall,
    fz0_loss,
    hit_indicators,
    kupiec_pof,
    var_backtest,
)
from volbench.dist import Distribution, Empirical, Normal, QuantileGrid, StudentT
from volbench.evaluate import run_backtest
from volbench.splitter import RollingOriginSplitter


def _ln0(x: float, y: float) -> float:
    """x * ln(y) with 0 * ln(0) = 0, written out for the hand transcriptions."""
    return 0.0 if x == 0 else x * math.log(y)


# --------------------------------------------------------------------------
# hits
# --------------------------------------------------------------------------


def test_hit_indicators_are_strict_and_propagate_nan() -> None:
    hits = hit_indicators([-0.02, -0.01, 0.0, np.nan, 0.01], [-0.01, -0.01, -0.01, -0.01, np.nan])
    assert hits.tolist()[:3] == [1.0, 0.0, 0.0]  # equality is not a hit, as in evaluate.py
    assert math.isnan(hits[3]) and math.isnan(hits[4])
    with pytest.raises(ValueError, match="same shape"):
        hit_indicators([0.0, 1.0], [0.0])


def test_hit_validation() -> None:
    with pytest.raises(ValueError, match="0/1"):
        kupiec_pof([0.0, 2.0, 1.0], 0.05)
    with pytest.raises(ValueError, match="1-D"):
        kupiec_pof(np.zeros((2, 2)), 0.05)
    with pytest.raises(ValueError, match="at least one"):
        kupiec_pof([np.nan, np.nan], 0.05)
    with pytest.raises(ValueError, match="level"):
        kupiec_pof([0.0, 1.0], 1.5)


# --------------------------------------------------------------------------
# Kupiec (1995)
# --------------------------------------------------------------------------


def test_kupiec_matches_hand_formula() -> None:
    n, x, level = 250, 5, 0.01
    hits = np.zeros(n)
    hits[[3, 40, 41, 100, 249]] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SmallSampleWarning)
        result = kupiec_pof(hits, level)
    assert isinstance(result, KupiecResult)
    lr = -2.0 * (_ln0(n - x, 1 - level) + _ln0(x, level)) + 2.0 * (
        _ln0(n - x, 1 - x / n) + _ln0(x, x / n)
    )
    assert result.lr == pytest.approx(lr, rel=1e-12)
    assert result.p_value == pytest.approx(stats.chi2.sf(lr, 1), rel=1e-12)
    assert (result.n, result.n_hits, result.hit_rate) == (n, x, x / n)
    assert result.expected_hits == pytest.approx(2.5)
    assert result.small_sample and result.n_dropped == 0


def test_kupiec_edge_cases_follow_the_zero_log_zero_convention() -> None:
    n, level = 500, 0.05
    none = kupiec_pof(np.zeros(n), level)
    assert none.lr == pytest.approx(-2.0 * n * math.log(1 - level))
    every = kupiec_pof(np.ones(n), level)
    assert every.lr == pytest.approx(-2.0 * n * math.log(level))
    exact = kupiec_pof(np.r_[np.ones(25), np.zeros(475)], level)  # x/n == level exactly
    assert exact.lr == 0.0 and exact.p_value == 1.0


def test_kupiec_size_under_correct_coverage() -> None:
    """3 000 seeded Bernoulli(0.05) sequences of n=1 000: rejection at 5% near nominal.

    The LR statistic is discrete, so the rejection rate is not exactly 0.05
    even asymptotically; the documented band is [0.03, 0.075] (MC SE 0.004;
    measured 0.051).
    """
    rng = np.random.default_rng(1)
    rejections = sum(kupiec_pof(rng.random(1000) < 0.05, 0.05).p_value < 0.05 for _ in range(3000))
    assert 0.03 < rejections / 3000 < 0.075, rejections / 3000


def test_kupiec_rejects_miscoverage_with_enough_hits() -> None:
    rng = np.random.default_rng(2)
    hits = (rng.random(2000) < 0.03).astype(float)  # 3% hits against a 1% VaR
    assert kupiec_pof(hits, 0.01).p_value < 1e-6


def test_small_sample_warning_carries_n_and_expected_hits() -> None:
    hits = np.zeros(250)
    with pytest.warns(SmallSampleWarning, match=r"n=250 at level 0.01 gives 2.50 expected"):
        result = kupiec_pof(hits, 0.01)
    assert result.small_sample and result.expected_hits == 2.5 and result.n == 250
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = kupiec_pof(hits, 0.01, warn=False)  # flag still set, no warning
        big = kupiec_pof(np.zeros(2000), 0.01)  # 20 expected hits: no warning
    assert quiet.small_sample and not big.small_sample
    assert MIN_EXPECTED_HITS == 10.0


def test_kupiec_drops_nan_hits_and_counts_them() -> None:
    hits = np.array([0.0, np.nan, 1.0, 0.0, np.nan, 0.0])
    result = kupiec_pof(hits, 0.25, warn=False)
    assert (result.n, result.n_hits, result.n_dropped) == (4, 1, 2)
    assert result.lr == pytest.approx(kupiec_pof([0.0, 1.0, 0.0, 0.0], 0.25, warn=False).lr)


# --------------------------------------------------------------------------
# Christoffersen (1998)
# --------------------------------------------------------------------------


def _hand_markov(hits: list[float], level: float) -> dict[str, float]:
    counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    for prev, cur in pairwise(hits):
        counts[f"{int(prev)}{int(cur)}"] += 1
    n00, n01, n10, n11 = counts["00"], counts["01"], counts["10"], counts["11"]
    total = n00 + n01 + n10 + n11
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / total
    ll_level = _ln0(n00 + n10, 1 - level) + _ln0(n01 + n11, level)
    ll_pi = _ln0(n00 + n10, 1 - pi) + _ln0(n01 + n11, pi)
    ll_markov = _ln0(n00, 1 - pi01) + _ln0(n01, pi01) + _ln0(n10, 1 - pi11) + _ln0(n11, pi11)
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_uc": -2 * (ll_level - ll_pi),
        "lr_ind": -2 * (ll_pi - ll_markov),
        "lr_cc": -2 * (ll_level - ll_markov),
    }


def test_christoffersen_matches_hand_transcription() -> None:
    hits = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
    level = 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SmallSampleWarning)
        result = christoffersen(hits, level)
    assert isinstance(result, ChristoffersenResult)
    hand = _hand_markov([float(h) for h in hits], level)
    assert (result.n00, result.n01, result.n10, result.n11) == (
        hand["n00"],
        hand["n01"],
        hand["n10"],
        hand["n11"],
    )
    assert result.n_transitions == 19 and result.n == 20 and result.n_hits == 7
    assert result.lr_uc == pytest.approx(hand["lr_uc"], rel=1e-12)
    assert result.lr_ind == pytest.approx(hand["lr_ind"], rel=1e-12)
    assert result.lr_cc == pytest.approx(hand["lr_cc"], rel=1e-12)
    assert result.lr_cc == pytest.approx(result.lr_uc + result.lr_ind, rel=1e-12)  # exact identity
    assert result.p_ind == pytest.approx(stats.chi2.sf(result.lr_ind, 1))
    assert result.p_cc == pytest.approx(stats.chi2.sf(result.lr_cc, 2))
    assert result.pi01 == pytest.approx(hand["n01"] / (hand["n00"] + hand["n01"]))
    assert result.pi11 == pytest.approx(hand["n11"] / (hand["n10"] + hand["n11"]))


def test_christoffersen_without_any_hit() -> None:
    result = christoffersen(np.zeros(300), 0.05)
    assert result.lr_ind == 0.0 and result.p_ind == 1.0
    assert result.pi01 == 0.0 and result.pi11 == 0.0  # undefined pi11 contributes nothing
    assert result.lr_uc == pytest.approx(-2.0 * 299 * math.log(0.95))  # n-1 transitions
    assert result.lr_cc == pytest.approx(result.lr_uc)


def test_christoffersen_detects_clustering_and_accepts_independence() -> None:
    rng = np.random.default_rng(3)
    n = 3000
    clustered = np.zeros(n)
    for t in range(1, n):
        p = 0.6 if clustered[t - 1] == 1.0 else 0.02
        clustered[t] = float(rng.random() < p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SmallSampleWarning)
        result = christoffersen(clustered, 0.05)
    assert result.pi11 > result.pi01
    assert result.p_ind < 1e-6 and result.p_cc < 1e-6
    independent = christoffersen((rng.random(n) < 0.05).astype(float), 0.05)
    assert independent.p_ind > 0.05  # seeded: 0.60 (p_uc is not asserted; coverage is Kupiec's job)


def test_christoffersen_only_pairs_adjacent_usable_observations() -> None:
    hits = np.array([0.0, np.nan, 1.0, 1.0, 0.0])
    result = christoffersen(hits, 0.4, warn=False)
    assert (result.n, result.n_dropped, result.n_transitions) == (4, 1, 2)
    assert (result.n00, result.n01, result.n10, result.n11) == (0, 0, 1, 1)
    with pytest.raises(ValueError, match="no adjacent pair"):
        christoffersen([0.0, np.nan, 1.0], 0.4)
    with pytest.raises(ValueError, match="at least two"):
        christoffersen([1.0], 0.4)


# --------------------------------------------------------------------------
# FZ0 (Patton, Ziegel & Chen 2019, eq. 6)
# --------------------------------------------------------------------------


def _fz0_scalar(y: float, v: float, e: float, a: float) -> float:
    """Eq. (6), transcribed term by term."""
    indicator = 1.0 if y <= v else 0.0
    return -(1.0 / (a * e)) * indicator * (v - y) + v / e + math.log(-e) - 1.0


def test_fz0_at_the_papers_figure_1_configuration() -> None:
    """Figure 1 of PZC: Y = -1, alpha = 0.05, (v, e) = (-1.64, -2.06)."""
    v, e, a = -1.64, -2.06, 0.05
    no_hit = fz0_loss(-1.0, v, e, a)
    assert no_hit == pytest.approx(v / e + math.log(-e) - 1.0)  # 0.51883...
    assert no_hit == pytest.approx(0.518822, abs=1e-6)  # 1.64/2.06 + ln 2.06 - 1
    hit = fz0_loss(-2.0, v, e, a)
    assert hit == pytest.approx(_fz0_scalar(-2.0, v, e, a))
    assert hit == pytest.approx(-(1.0 / (a * e)) * (v + 2.0) + no_hit)
    assert fz0_loss(v, v, e, a) == pytest.approx(no_hit)  # Y == v: indicator on, zero shortfall


def test_fz0_vectorised_equals_scalar_transcription() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(size=200)
    v = -np.abs(rng.normal(size=200)) - 0.5
    e = v - np.abs(rng.normal(size=200)) - 0.1
    losses = fz0_loss(y, v, e, 0.025)
    expected = np.array([_fz0_scalar(*args, 0.025) for args in zip(y, v, e, strict=True)])
    np.testing.assert_allclose(losses, expected, rtol=1e-12)


def test_fz0_scale_identity_from_pzc_eq_42() -> None:
    """L(kY, kv, ke) = L(Y, v, e) + log k for k > 0: loss differences are 0-homogeneous."""
    rng = np.random.default_rng(5)
    y = rng.normal(size=50)
    v = -np.abs(rng.normal(size=50)) - 0.2
    e = v - 0.3
    for k in (0.01, 0.5, 3.0):
        scaled = fz0_loss(k * y, k * v, k * e, 0.05)
        np.testing.assert_allclose(scaled, fz0_loss(y, v, e, 0.05) + math.log(k), rtol=1e-10)


def test_fz0_domain_is_enforced() -> None:
    with pytest.raises(ValueError, match="strictly below zero"):
        fz0_loss([0.1], [-1.0], [0.0], 0.05)
    with pytest.raises(ValueError, match="strictly below zero"):
        fz0_loss([0.1], [1.0], [2.0], 0.05)  # loss-side (positive) convention: refused
    with pytest.raises(ValueError, match="ES <= VaR"):
        fz0_loss([0.1], [-2.0], [-1.0], 0.05)
    with pytest.raises(ValueError, match="level"):
        fz0_loss([0.1], [-1.0], [-2.0], 0.0)
    assert math.isfinite(fz0_loss(0.1, 0.2, -0.5, 0.05))  # positive VaR, negative ES: allowed


def test_fz0_propagates_nan_positionally() -> None:
    losses = fz0_loss([np.nan, -1.0, -1.0], [-1.0, np.nan, -1.0], [-2.0, -2.0, np.inf], 0.05)
    assert np.isnan(losses).all()
    mixed = fz0_loss([np.nan, -1.0], [-1.0, -1.0], [-2.0, -2.0], 0.05)
    assert math.isnan(mixed[0]) and math.isfinite(mixed[1])


def _expected_fz0_standard_normal(v: float, e: float, a: float) -> float:
    """E[L_FZ0] under Y ~ N(0, 1): E[1{Y<=v}(v-Y)] = v Phi(v) + phi(v)."""
    partial = v * stats.norm.cdf(v) + stats.norm.pdf(v)
    return float(-partial / (a * e) + v / e + math.log(-e) - 1.0)


def test_fz0_expected_loss_is_minimised_at_the_true_var_and_es() -> None:
    """PZC Figure 2: for N(0,1) at alpha=0.05 the minimum sits at (-1.645, -2.063)."""
    a = 0.05
    true_v = float(stats.norm.ppf(a))
    true_e = -float(stats.norm.pdf(true_v)) / a
    assert true_v == pytest.approx(-1.6449, abs=1e-4) and true_e == pytest.approx(-2.0627, abs=1e-4)
    grid_v = np.linspace(-2.2, -1.1, 45)
    grid_e = np.linspace(-2.8, -1.6, 49)
    best = min(
        (_expected_fz0_standard_normal(v, e, a), v, e) for v in grid_v for e in grid_e if e <= v
    )
    assert best[1] == pytest.approx(true_v, abs=0.03)
    assert best[2] == pytest.approx(true_e, abs=0.03)
    assert _expected_fz0_standard_normal(true_v, true_e, a) < _expected_fz0_standard_normal(
        true_v * 1.3, true_e * 1.3, a
    )
    # And the Monte Carlo mean of fz0_loss agrees with the closed-form expectation.
    y = np.random.default_rng(6).normal(size=400_000)
    mc = float(fz0_loss(y, true_v, true_e, a).mean())
    assert mc == pytest.approx(_expected_fz0_standard_normal(true_v, true_e, a), abs=0.01)


def test_fz0_ranks_the_correct_forecast_first() -> None:
    rng = np.random.default_rng(7)
    sigma, a = 0.012, 0.01
    y = rng.normal(0.0, sigma, size=60_000)
    truth = Normal(0.0, sigma)
    scores = {}
    for label, scale in (("right", 1.0), ("wide", 1.6), ("narrow", 0.6)):
        dist = Normal(0.0, sigma * scale)
        scores[label] = float(fz0_loss(y, dist.quantile(a), expected_shortfall(dist, a), a).mean())
    assert scores["right"] < min(scores["wide"], scores["narrow"])
    assert truth.quantile(a) > expected_shortfall(truth, a)


# --------------------------------------------------------------------------
# expected shortfall
# --------------------------------------------------------------------------


class _QuantileOnly(Distribution):
    """A distribution exposing nothing but ``quantile``: exercises the quadrature path."""

    def __init__(self, inner: Distribution) -> None:
        self.inner = inner

    def quantile(self, tau: float) -> float:
        return self.inner.quantile(tau)


@pytest.mark.parametrize("level", [0.01, 0.025, 0.05, 0.1])
def test_es_normal_closed_form_and_generic_quadrature_agree(level: float) -> None:
    dist = Normal(0.0004, 0.011)
    closed = expected_shortfall(dist, level)
    z = stats.norm.ppf(level)
    assert closed == pytest.approx(0.0004 - 0.011 * stats.norm.pdf(z) / level, rel=1e-12)
    numeric = integrate.quad(lambda u: dist.quantile(u), 0.0, level, limit=200)[0] / level
    assert closed == pytest.approx(numeric, rel=1e-8)
    assert expected_shortfall(_QuantileOnly(dist), level) == pytest.approx(closed, rel=1e-6)
    assert closed < dist.quantile(level)


@pytest.mark.parametrize("df", [2.5, 4.0, 8.0, 30.0])
def test_es_student_t_closed_form_against_numerical_integration(df: float) -> None:
    dist = StudentT.from_variance(0.0, 0.012**2, df)
    level = 0.025
    closed = expected_shortfall(dist, level)
    numeric = integrate.quad(lambda u: dist.quantile(u), 0.0, level, limit=400)[0] / level
    assert closed == pytest.approx(numeric, rel=1e-7)
    assert closed < dist.quantile(level)
    assert expected_shortfall(_QuantileOnly(dist), level) == pytest.approx(closed, rel=1e-5)


def test_es_empirical_is_exact_for_numpys_linear_quantile_function() -> None:
    samples = np.sort(np.random.default_rng(8).standard_t(4, size=97))
    dist = Empirical(samples)
    level = 0.06
    fine = np.linspace(0.0, level, 200_001)
    brute = float(np.trapezoid(np.quantile(samples, fine), fine)) / level
    assert expected_shortfall(dist, level) == pytest.approx(brute, rel=1e-6)
    assert expected_shortfall(Empirical(np.array([-1.5])), level) == -1.5


def test_es_quantile_grid_is_exact_for_the_interpolant() -> None:
    dist = QuantileGrid(np.array([0.01, 0.05, 0.5]), np.array([-3.0, -2.0, 0.0]))
    # int_0^0.05 Q = 0.01 * (-3) + 0.04 * (-2.5)  ->  ES = -0.13 / 0.05 = -2.6
    assert expected_shortfall(dist, 0.05) == pytest.approx(-2.6)
    assert expected_shortfall(dist, 0.005) == pytest.approx(-3.0)  # flat below the first tau
    assert expected_shortfall(dist, 0.03) == pytest.approx((0.01 * -3.0 + 0.02 * -2.75) / 0.03)


def test_es_level_validation() -> None:
    with pytest.raises(ValueError, match="level"):
        expected_shortfall(Normal(0.0, 1.0), 1.0)


# --------------------------------------------------------------------------
# from result rows
# --------------------------------------------------------------------------

SIGMA = 0.012


@dataclass(frozen=True)
class _OracleFit:
    sigma: float

    def predict(self, h: int) -> Distribution:
        return Distribution.from_normal(0.0, self.sigma)


@dataclass(frozen=True)
class Oracle:
    sigma: float
    label: str

    @property
    def name(self) -> str:
        return self.label

    def spec(self) -> dict[str, Any]:
        return {"kind": "oracle", "sigma": self.sigma}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _OracleFit:
        return _OracleFit(self.sigma)


def _cell(scale: float, label: str, *, proxy_nan_at: tuple[int, ...] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = 1200
    series = rng.normal(0.0, SIGMA, size=n)
    proxy = series**2 + 1e-8
    for i in proxy_nan_at:
        proxy[i] = np.nan
    splitter = RollingOriginSplitter(window=200, horizon=1, step=1, refit_every=50)
    return run_backtest(
        lambda: Oracle(SIGMA * scale, label),
        series,
        proxy,
        splitter,
        seed=1,
        asset="SIM",
        proxy_name="r2",
    )


def _es_column(frame: pd.DataFrame, level: float) -> NDArray[np.float64]:
    """ES implied by each row's recorded Normal forecast (the oracle is Normal)."""
    return np.array(
        [
            expected_shortfall(Normal(m, math.sqrt(v)), level)
            for m, v in zip(frame["forecast_mean"], frame["forecast_var"], strict=True)
        ]
    )


def test_var_backtest_on_scored_rows() -> None:
    level = 0.05
    right = _cell(1.0, "right")
    wide = _cell(3.0, "wide")
    n = len(right)
    assert n == 1000
    ok = var_backtest(right, level, es=_es_column(right, level))
    assert isinstance(ok, VaRBacktest)
    assert ok.n == n and ok.n_dropped == 0 and ok.expected_hits == pytest.approx(50.0)
    assert not ok.small_sample
    assert abs(ok.hit_rate - level) < 0.025
    assert ok.kupiec.p_value > 0.01 and ok.christoffersen.p_cc > 0.01
    assert ok.fz0_n == n and ok.fz0_mean is not None
    bad = var_backtest(wide, level, es=_es_column(wide, level))
    assert bad.n_hits <= 2  # a 3x-too-wide forecast is almost never breached
    assert bad.kupiec.p_value < 1e-4 and bad.christoffersen.p_cc < 1e-3
    assert bad.fz0_mean is not None and ok.fz0_mean < bad.fz0_mean
    without_es = var_backtest(right, level)
    assert without_es.fz0_mean is None and without_es.fz0_n == 0
    assert without_es.kupiec == ok.kupiec
    assert len(ok.config_hash) == 64 and ok.config_hash != without_es.config_hash
    shuffled = right.sample(frac=1.0, random_state=0)
    assert var_backtest(shuffled, level).christoffersen == without_es.christoffersen


def test_var_backtest_missing_policy_and_bookkeeping() -> None:
    level = 0.05
    cell = _cell(1.0, "right", proxy_nan_at=(500, 501))
    assert (cell["missing_reason"] == "proxy_nan").sum() == 2
    flagged = var_backtest(cell, level)
    assert flagged.n_dropped == 2 and flagged.n == 998
    assert flagged.kupiec.n_dropped == 2 and flagged.christoffersen.n_dropped == 2
    by_score = var_backtest(cell, level, policy="score")
    assert by_score.n_dropped == 0 and by_score.n == 1000
    assert flagged.config_hash != by_score.config_hash


def test_var_backtest_warns_on_small_samples_once() -> None:
    cell = _cell(1.0, "right").iloc[:250]
    with pytest.warns(SmallSampleWarning, match="n=250 at level 0.01") as record:
        result = var_backtest(cell, 0.01)
    assert len(record) == 1
    assert result.small_sample and result.expected_hits == pytest.approx(2.5)


def test_var_backtest_validation() -> None:
    right, wide = _cell(1.0, "right"), _cell(3.0, "wide")
    with pytest.raises(ValueError, match="exactly one cell"):
        var_backtest(pd.concat([right, wide]), 0.05)
    with pytest.raises(ValueError, match="scored levels"):
        var_backtest(right, 0.03)
    with pytest.raises(ValueError, match="one value per row"):
        var_backtest(right, 0.05, es=np.zeros(3))
    with pytest.raises(ValueError, match="es column"):
        var_backtest(right, 0.05, es="es_0p05")
    with pytest.raises(ValueError, match="policy"):
        var_backtest(right, 0.05, policy="drop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        var_backtest(right.iloc[:0], 0.05)
    with pytest.raises(ValueError, match="duplicate origin_index"):
        var_backtest(pd.concat([right, right]), 0.05)
