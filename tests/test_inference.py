"""Comparison inference: Diebold-Mariano / HLN, the moving block bootstrap, MCS.

Three load-bearing checks:

- ``test_hln_at_h1_is_exactly_the_one_sample_t_test`` — the HLN-corrected DM
  statistic at h=1 is algebraically the one-sample t statistic, so any error
  in the variance estimator, the correction factor or the reference
  distribution shows up here.
- ``test_dm_size_iid_gaussian`` — the size check the Phase 2 brief demands.
- ``test_block_bootstrap_respects_time_order`` — every block is a contiguous,
  forward-running window of the original axis (the leakage-check item for
  this module).

Every train/test index in the backtest-based tests comes from
``RollingOriginSplitter``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy import stats

from volbench.dist import Distribution
from volbench.evaluate import run_backtest
from volbench.inference import (
    DEFAULT_ALPHA,
    DEFAULT_N_BOOT,
    DMMatrix,
    LossMatrix,
    MCSResult,
    _bootstrap_column_means,
    _ppw_block_length,
    compare_models,
    default_block_length,
    diebold_mariano,
    dm_matrix,
    loss_matrix,
    model_confidence_set,
    moving_block_indices,
)
from volbench.results import ResultsStore
from volbench.splitter import RollingOriginSplitter

# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------


def test_hln_at_h1_is_exactly_the_one_sample_t_test() -> None:
    """HLN (1997, eq. 9) at h=1: sqrt((n-1)/n) * dbar/sqrt(gamma_0/n) == t-statistic.

    gamma_0 is the 1/n autocovariance, so the factor converts it to the
    unbiased 1/(n-1) variance and the statistic is scipy's one-sample t,
    compared against t_{n-1}. Exact, not approximate.
    """
    rng = np.random.default_rng(1)
    loss_a = rng.gamma(2.0, size=80)
    loss_b = rng.gamma(2.0, size=80)
    result = diebold_mariano(loss_a, loss_b)
    reference = stats.ttest_1samp(loss_a - loss_b, 0.0)
    assert result.statistic == pytest.approx(float(reference.statistic), rel=1e-12)
    assert result.p_value == pytest.approx(float(reference.pvalue), rel=1e-12)
    assert result.n == 80
    assert result.n_dropped == 0
    assert result.hln is True
    assert result.lag == 0


def test_original_dm_statistic_without_correction() -> None:
    rng = np.random.default_rng(2)
    d = rng.normal(size=60)
    result = diebold_mariano(d, np.zeros_like(d), hln=False)
    gamma0 = float(np.mean((d - d.mean()) ** 2))
    expected = d.mean() / math.sqrt(gamma0 / d.size)
    assert result.statistic == pytest.approx(expected, rel=1e-12)
    assert result.p_value == pytest.approx(2.0 * stats.norm.sf(abs(expected)), rel=1e-12)


def _hand_dm(d: NDArray[np.float64], h: int, bartlett: bool) -> tuple[float, float]:
    """Independent, loop-based transcription of DM (1995) + HLN (1997, eq. 9)."""
    n = d.size
    dbar = d.mean()
    gammas = []
    for k in range(h):
        acc = 0.0
        for t in range(k, n):
            acc += (d[t] - dbar) * (d[t - k] - dbar)
        gammas.append(acc / n)
    var = gammas[0]
    for k in range(1, h):
        w = 1.0 - k / h if bartlett else 1.0
        var += 2.0 * w * gammas[k]
    var /= n
    s1 = dbar / math.sqrt(var)
    factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return factor * s1, var


@pytest.mark.parametrize("kernel", ["rectangular", "bartlett"])
def test_multi_step_variance_and_factor_match_hand_transcription(kernel: str) -> None:
    rng = np.random.default_rng(3)
    eps = rng.normal(size=120)
    d = eps[2:] + eps[1:-1] + eps[:-2]  # MA(2): what a 3-step loss differential carries
    result = diebold_mariano(d + 0.1, np.zeros_like(d), horizon=3, kernel=kernel)  # type: ignore[arg-type]
    statistic, variance = _hand_dm(d + 0.1, 3, kernel == "bartlett")
    assert result.lag == 2
    assert result.variance == pytest.approx(variance, rel=1e-12)
    assert result.statistic == pytest.approx(statistic, rel=1e-12)
    assert result.p_value == pytest.approx(2.0 * stats.t.sf(abs(statistic), df=d.size - 1))


def test_lag_override_moves_the_hln_factor_with_it() -> None:
    rng = np.random.default_rng(4)
    d = rng.normal(size=100)
    explicit = diebold_mariano(d, np.zeros_like(d), horizon=1, lag=4)
    implied = diebold_mariano(d, np.zeros_like(d), horizon=5)
    assert explicit.statistic == pytest.approx(implied.statistic)
    assert explicit.lag == implied.lag == 4


def test_dm_size_iid_gaussian() -> None:
    """Size check (Phase 2 brief): iid zero-mean differentials reject at ~nominal.

    4 000 seeded replications of n=40. At h=1 the corrected test is exact
    under Gaussian iid differentials, so the only slack is Monte Carlo error:
    SE = sqrt(a(1-a)/4000) = 0.0034 at a=0.05 and 0.0047 at a=0.10. The
    tolerance is 3.5 SE, i.e. +-0.012 and +-0.017. Seeded, so the check is
    deterministic (measured: 0.043 and 0.096); the tolerance says what a
    re-seeded run may do.
    """
    rng = np.random.default_rng(20260825)
    reps, n = 4000, 40
    p_values = np.array(
        [diebold_mariano(rng.normal(size=n), np.zeros(n)).p_value for _ in range(reps)]
    )
    for alpha in (0.05, 0.10):
        rate = float(np.mean(p_values < alpha))
        tolerance = 3.5 * math.sqrt(alpha * (1.0 - alpha) / reps)
        assert abs(rate - alpha) < tolerance, f"alpha={alpha}: rejection rate {rate:.4f}"


def test_dm_size_serially_correlated_h3() -> None:
    """MA(2) differentials, h=3, rectangular window: size close to nominal.

    HLN (1997, Table 1) show the corrected test is still mildly over-sized in
    small samples for h > 1; at n=200 the documented band here is
    [0.03, 0.09] around a nominal 0.05 (2 000 replications, MC SE 0.005;
    measured 0.054).
    """
    rng = np.random.default_rng(7)
    reps, n = 2000, 200
    rejections = 0
    for _ in range(reps):
        eps = rng.normal(size=n + 2)
        d = eps[2:] + eps[1:-1] + eps[:-2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rejections += diebold_mariano(d, np.zeros(n), horizon=3).p_value < 0.05
    rate = rejections / reps
    assert 0.03 < rate < 0.09, rate


def test_dm_has_power() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(loc=0.5, size=200)
    b = rng.normal(size=200)
    result = diebold_mariano(a, b)
    assert result.p_value < 1e-3
    assert result.statistic > 0  # a lost more
    assert diebold_mariano(a, b, alternative="greater").p_value < 1e-3
    assert diebold_mariano(a, b, alternative="less").p_value > 0.99


def test_one_sided_p_values_are_complementary() -> None:
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=50), rng.normal(size=50)
    two = diebold_mariano(a, b).p_value
    less = diebold_mariano(a, b, alternative="less").p_value
    greater = diebold_mariano(a, b, alternative="greater").p_value
    assert less + greater == pytest.approx(1.0)
    assert two == pytest.approx(2.0 * min(less, greater))


def test_nan_policy_is_pairwise_complete_and_counted() -> None:
    rng = np.random.default_rng(8)
    a = rng.normal(size=60)
    b = rng.normal(size=60)
    a[[3, 10]] = np.nan
    b[[10, 20, 41]] = np.inf
    result = diebold_mariano(a, b)
    keep = np.isfinite(a) & np.isfinite(b)
    reference = diebold_mariano(a[keep], b[keep])
    assert result.n == 56
    assert result.n_dropped == 4
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.p_value)


def test_identical_losses_are_no_evidence() -> None:
    loss = np.random.default_rng(9).gamma(1.0, size=30)
    result = diebold_mariano(loss, loss.copy())
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.variance == 0.0
    assert not result.variance_nonpositive


def test_negative_rectangular_variance_applies_the_dm_rule() -> None:
    """DM (1995, §1.1): a non-positive spectral estimate is set to 0 and the null rejected."""
    n = 40
    alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    # mean exactly zero: nothing to reject, but the estimate is still negative.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zero_mean = diebold_mariano(alternating, np.zeros(n), horizon=2)
    assert zero_mean.variance < 0
    assert zero_mean.statistic == 0.0
    assert zero_mean.p_value == 1.0
    assert zero_mean.variance_nonpositive
    # non-zero mean: rejected outright, flagged and warned.
    with pytest.warns(RuntimeWarning, match="treating it as 0"):
        shifted = diebold_mariano(alternating + 0.1, np.zeros(n), horizon=2)
    assert shifted.variance_nonpositive
    assert shifted.statistic == math.inf
    assert shifted.p_value == 0.0
    # The Bartlett window is positive on the same data.
    bartlett = diebold_mariano(alternating + 0.1, np.zeros(n), horizon=2, kernel="bartlett")
    assert bartlett.variance > 0
    assert not bartlett.variance_nonpositive


def test_dm_config_hash_tracks_inputs_and_settings() -> None:
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=30), rng.normal(size=30)
    base = diebold_mariano(a, b)
    assert base.config_hash == diebold_mariano(a.copy(), b.copy()).config_hash
    assert base.config_hash != diebold_mariano(b, a).config_hash
    assert base.config_hash != diebold_mariano(a, b, hln=False).config_hash
    assert base.config_hash != diebold_mariano(a, b, horizon=2).config_hash
    assert base.config_hash != diebold_mariano(a, b, kernel="bartlett", horizon=2).config_hash


def test_dm_input_validation() -> None:
    a = np.ones(10)
    with pytest.raises(ValueError, match="length"):
        diebold_mariano(a, np.ones(9))
    with pytest.raises(ValueError, match="complete loss pairs"):
        diebold_mariano(np.array([1.0, math.nan, 2.0]), np.array([1.0, 2.0, math.nan]))
    with pytest.raises(ValueError, match="complete loss pairs"):
        diebold_mariano(np.arange(5.0), np.zeros(5), horizon=5)
    with pytest.raises(ValueError, match="kernel"):
        diebold_mariano(a, a, kernel="parzen")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alternative"):
        diebold_mariano(a, a, alternative="different")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1-D"):
        diebold_mariano(np.ones((2, 5)), np.ones((2, 5)))


def test_dm_matrix_orientation_and_bookkeeping() -> None:
    rng = np.random.default_rng(11)
    losses = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    losses.loc[5, "c"] = np.nan
    matrix = dm_matrix(losses)
    assert isinstance(matrix, DMMatrix)
    assert matrix.models == ("a", "b", "c")
    s, p = matrix.statistic, matrix.p_value
    assert np.isnan(np.diag(s)).all() and np.isnan(np.diag(p)).all()
    assert s.loc["a", "b"] == pytest.approx(-s.loc["b", "a"])
    assert p.loc["a", "b"] == pytest.approx(p.loc["b", "a"])
    assert int(matrix.n.loc["a", "b"]) == 80 and int(matrix.n_dropped.loc["a", "b"]) == 0
    assert int(matrix.n.loc["a", "c"]) == 79 and int(matrix.n_dropped.loc["a", "c"]) == 1
    pair = matrix.pairs[("a", "c")]
    assert pair.n == 79 and pair.statistic == pytest.approx(s.loc["a", "c"])
    direct = diebold_mariano(losses["a"].to_numpy(), losses["b"].to_numpy())
    assert s.loc["a", "b"] == pytest.approx(direct.statistic)


# --------------------------------------------------------------------------
# moving block bootstrap
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "block_length"), [(10, 1), (10, 3), (10, 4), (10, 10), (37, 5)])
def test_block_bootstrap_respects_time_order(n: int, block_length: int) -> None:
    """Every block is s, s+1, ..., s+l-1 of the ORIGINAL axis, never wrapping.

    This is the leakage-check requirement for this module: resampled time
    runs forward inside every block, so the serial dependence the method
    relies on is the sample's own, and no block joins the end of the sample
    to its start.
    """
    indices = moving_block_indices(n, block_length, 50, seed=0)
    assert indices.shape == (50, n)
    assert indices.min() >= 0 and indices.max() < n
    for row in indices:
        for start in range(0, n, block_length):
            block = row[start : start + block_length]
            assert block[0] <= n - block_length  # a legal window start: no wrap-around
            assert np.array_equal(block, block[0] + np.arange(block.size))  # forward, contiguous


def test_block_length_n_reproduces_the_sample_and_length_1_is_iid() -> None:
    identity = moving_block_indices(12, 12, 5, seed=1)
    assert np.array_equal(identity, np.tile(np.arange(12), (5, 1)))
    iid = moving_block_indices(12, 1, 200, seed=1)
    assert set(np.unique(iid)) == set(range(12))
    assert not np.array_equal(iid[0], np.sort(iid[0]))  # not the identity ordering


def test_block_bootstrap_is_seeded() -> None:
    assert np.array_equal(
        moving_block_indices(30, 4, 20, seed=3), moving_block_indices(30, 4, 20, seed=3)
    )
    assert not np.array_equal(
        moving_block_indices(30, 4, 20, seed=3), moving_block_indices(30, 4, 20, seed=4)
    )


@pytest.mark.parametrize(
    ("n", "block_length", "n_boot"), [(50, 1, 700), (50, 7, 300), (50, 50, 3), (9, 4, 600)]
)
def test_block_sum_means_equal_explicit_index_means(n: int, block_length: int, n_boot: int) -> None:
    """The internal cumulative-sum path and the public index generator agree exactly.

    n_boot spans several RNG chunks so the chunked draws are covered.
    """
    rng = np.random.default_rng(12)
    values = rng.normal(size=(n, 3))
    via_sums = _bootstrap_column_means(values, block_length, n_boot, np.random.default_rng(99))
    indices = moving_block_indices(n, block_length, n_boot, seed=99)
    via_index = values[indices].mean(axis=1)
    assert via_sums.shape == (n_boot, 3)
    np.testing.assert_allclose(via_sums, via_index, rtol=1e-12, atol=1e-12)


def test_block_bootstrap_validation() -> None:
    with pytest.raises(ValueError, match="block_length"):
        moving_block_indices(10, 0, 5, seed=0)
    with pytest.raises(ValueError, match="block_length"):
        moving_block_indices(10, 11, 5, seed=0)
    with pytest.raises(ValueError, match="n_boot"):
        moving_block_indices(10, 2, 0, seed=0)


# --------------------------------------------------------------------------
# block length rule
# --------------------------------------------------------------------------


def _ar1(rng: np.random.Generator, phi: float, n: int) -> NDArray[np.float64]:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def test_politis_white_block_length_matches_the_reference_implementation() -> None:
    """Pinned against ``arch.bootstrap.optimal_block_length`` (circular column), exactly."""
    from arch.bootstrap import optimal_block_length

    rng = np.random.default_rng(13)
    series = {
        "white": rng.normal(size=400),
        "ar1_0.5": _ar1(rng, 0.5, 400),
        "ar1_0.9": _ar1(rng, 0.9, 400),
        "ma2": np.convolve(rng.normal(size=402), np.ones(3), mode="valid"),
        "heavy": rng.standard_t(3, size=250) ** 2,
        "short": rng.normal(size=30),
    }
    for name, x in series.items():
        reference = float(optimal_block_length(x)["circular"].iloc[0])
        assert _ppw_block_length(x) == pytest.approx(reference, rel=1e-12), name


def test_politis_white_block_length_degenerate_inputs() -> None:
    assert _ppw_block_length(np.ones(100)) == 1.0  # constant: nothing to estimate
    assert math.isfinite(_ppw_block_length(np.random.default_rng(0).normal(size=12)))


def test_default_block_length_floors_at_horizon_and_grows_with_dependence() -> None:
    rng = np.random.default_rng(14)
    iid = rng.normal(size=(400, 3))
    assert 1 <= default_block_length(iid) <= 3
    assert default_block_length(iid, horizon=5) == 5
    persistent = iid.copy()
    persistent[:, 0] += 3.0 * _ar1(rng, 0.9, 400)  # one model's loss carries an AR(1) component
    assert default_block_length(persistent) >= 5
    assert default_block_length(persistent) == math.ceil(
        max(
            _ppw_block_length(persistent[:, 0] - persistent[:, 1]),
            _ppw_block_length(persistent[:, 0] - persistent[:, 2]),
            _ppw_block_length(persistent[:, 1] - persistent[:, 2]),
        )
    )
    assert default_block_length(iid[:2]) == 1  # too short to estimate: the floor


# --------------------------------------------------------------------------
# model confidence set
# --------------------------------------------------------------------------


@pytest.mark.parametrize("statistic", ["range", "max"])
def test_identical_losses_keep_every_model(statistic: str) -> None:
    """Correctness pin (Phase 2 brief): identical losses -> all in the set, p = 1."""
    rng = np.random.default_rng(15)
    common = rng.gamma(2.0, size=150)
    losses = np.column_stack([common, common, common, common])
    result = model_confidence_set(losses, seed=1, n_boot=500, statistic=statistic)  # type: ignore[arg-type]
    assert result.included == ("model_0", "model_1", "model_2", "model_3")
    assert result.excluded == ()
    assert all(p == 1.0 for p in result.p_values.values())
    assert all(p == 1.0 for p in result.step_p_values)


@pytest.mark.parametrize("statistic", ["range", "max"])
def test_dominated_model_is_eliminated_first(statistic: str) -> None:
    """Correctness pin (Phase 2 brief): one strictly dominated model goes first."""
    rng = np.random.default_rng(16)
    base = rng.normal(size=(300, 4))
    base[:, 2] += 1.0  # strictly worse in expectation, by one standard deviation
    names = ["good_a", "good_b", "bad", "good_c"]
    result = model_confidence_set(base, seed=2, n_boot=1000, model_names=names, statistic=statistic)  # type: ignore[arg-type]
    assert result.elimination_order[0] == "bad"
    assert result.p_values["bad"] < DEFAULT_ALPHA
    assert result.excluded == ("bad",)
    assert result.included == ("good_a", "good_b", "good_c")
    assert result.mean_loss["bad"] > max(result.mean_loss[n] for n in result.included)


def test_mcs_p_values_are_cumulative_maxima_with_survivor_at_one() -> None:
    rng = np.random.default_rng(17)
    losses = rng.normal(size=(120, 5)) + np.array([0.0, 0.05, 0.3, 0.6, 1.0])
    result = model_confidence_set(losses, seed=3, n_boot=800)
    assert isinstance(result, MCSResult)
    order = result.elimination_order
    assert len(order) == 5 and set(order) == set(result.models)
    mcs_p = [result.p_values[name] for name in order]
    assert mcs_p == sorted(mcs_p)  # non-decreasing along the elimination sequence
    assert mcs_p[-1] == 1.0
    running = 0.0
    for name, step in zip(order[:-1], result.step_p_values, strict=True):
        running = max(running, step)
        assert result.p_values[name] == running
    assert result.included == tuple(n for n in result.models if result.p_values[n] >= result.alpha)
    assert set(result.excluded) | set(result.included) == set(result.models)


def test_mcs_covers_the_true_set_at_least_1_minus_alpha() -> None:
    """Size: with all models equal, the MCS keeps all of them >= 1-alpha of the time.

    100 seeded replications (n=150, m=4, iid losses, B=300). At alpha=0.10
    the target frequency is 0.90 with MC SE 0.03; asserted at >= 0.82
    (about 2.7 SE below; measured 0.92). Documented as a size check, not a
    pin on a number.
    """
    rng = np.random.default_rng(18)
    kept_all = 0
    for rep in range(100):
        losses = rng.normal(size=(150, 4))
        result = model_confidence_set(losses, seed=rep, n_boot=300, block_length=1)
        kept_all += len(result.included) == 4
    assert kept_all / 100 >= 0.82, kept_all


def test_mcs_nan_policy_is_listwise_and_counted() -> None:
    rng = np.random.default_rng(19)
    losses = rng.normal(size=(100, 3))
    losses[[2, 50], 0] = np.nan
    losses[7, 2] = np.inf
    result = model_confidence_set(losses, seed=4, n_boot=200, block_length=2)
    keep = np.all(np.isfinite(losses), axis=1)
    reference = model_confidence_set(losses[keep], seed=4, n_boot=200, block_length=2)
    assert result.n == 97 and result.n_dropped == 3
    assert reference.n == 97 and reference.n_dropped == 0
    assert result.p_values == reference.p_values
    assert result.elimination_order == reference.elimination_order


def test_mcs_is_deterministic_under_seed() -> None:
    rng = np.random.default_rng(20)
    losses = rng.normal(size=(80, 3)) + np.array([0.0, 0.2, 0.8])
    first = model_confidence_set(losses, seed=5, n_boot=400)
    second = model_confidence_set(losses, seed=5, n_boot=400)
    assert first.p_values == second.p_values
    assert first.config_hash == second.config_hash
    other = model_confidence_set(losses, seed=6, n_boot=400)
    assert other.config_hash != first.config_hash
    assert other.included == first.included  # a clear-cut case survives re-seeding


def test_mcs_records_block_length_and_hash_moves_with_settings() -> None:
    rng = np.random.default_rng(21)
    losses = rng.normal(size=(60, 3))
    auto = model_confidence_set(losses, seed=1, n_boot=100)
    assert auto.block_length == default_block_length(losses)
    fixed = model_confidence_set(losses, seed=1, n_boot=100, block_length=4)
    assert fixed.block_length == 4
    assert fixed.config_hash != auto.config_hash
    assert (
        model_confidence_set(losses, seed=1, n_boot=100, alpha=0.2).config_hash != auto.config_hash
    )
    assert model_confidence_set(losses, seed=1, n_boot=101).config_hash != auto.config_hash


def test_mcs_default_settings_are_the_reference_ones() -> None:
    assert DEFAULT_ALPHA == 0.10
    assert DEFAULT_N_BOOT == 10_000
    rng = np.random.default_rng(22)
    losses = rng.normal(size=(200, 5)) + np.array([0.0, 0.0, 0.0, 0.5, 1.0])
    result = model_confidence_set(losses, seed=7)  # default B = 10_000 must be affordable
    assert result.n_boot == 10_000 and result.alpha == 0.10 and result.statistic == "range"
    assert result.elimination_order[0] == "model_4"


def test_mcs_validation() -> None:
    losses = np.random.default_rng(23).normal(size=(30, 2))
    with pytest.raises(ValueError, match="alpha"):
        model_confidence_set(losses, seed=0, alpha=1.5)
    with pytest.raises(ValueError, match="statistic"):
        model_confidence_set(losses, seed=0, statistic="mean")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two models"):
        model_confidence_set(losses[:, :1], seed=0)
    with pytest.raises(ValueError, match="block_length"):
        model_confidence_set(losses, seed=0, block_length=31)
    with pytest.raises(ValueError, match="distinct"):
        model_confidence_set(losses, seed=0, model_names=["x", "x"])
    with pytest.raises(ValueError, match="complete"):
        model_confidence_set(np.full((5, 2), np.nan), seed=0)


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
    """Forecasts a fixed N(0, sigma^2) whatever the data: no estimation noise."""

    sigma: float
    label: str

    @property
    def name(self) -> str:
        return self.label

    def spec(self) -> dict[str, Any]:
        return {"kind": "oracle", "sigma": self.sigma}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _OracleFit:
        return _OracleFit(self.sigma)


def _scored_rows(scales: dict[str, float], *, proxy_nan_at: tuple[int, ...] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    n = 360
    series = rng.normal(0.0, SIGMA, size=n)
    proxy = series**2 + 1e-8
    for i in proxy_nan_at:
        proxy[i] = np.nan
    splitter = RollingOriginSplitter(window=120, horizon=1, step=1, refit_every=20)
    frames = [
        run_backtest(
            lambda scale=scale, label=label: Oracle(SIGMA * scale, label),  # type: ignore[misc]
            series,
            proxy,
            splitter,
            seed=1,
            asset="SIM",
            proxy_name="r2",
        )
        for label, scale in scales.items()
    ]
    return pd.concat(frames, ignore_index=True)


def test_loss_matrix_from_backtest_rows() -> None:
    rows = _scored_rows({"right": 1.0, "wide": 3.0})
    matrix = loss_matrix(rows, "crps")
    assert isinstance(matrix, LossMatrix)
    assert matrix.models == ("right", "wide")
    assert matrix.asset == "SIM" and matrix.horizon == 1 and matrix.score == "crps"
    assert matrix.values.index.is_monotonic_increasing  # ascending time order
    assert list(matrix.values.index) == sorted(rows["origin_index"].unique())
    assert matrix.n_flagged == {"right": 0, "wide": 0}
    assert set(matrix.config_hashes) == {"right", "wide"}
    assert matrix.config_hashes["right"] != matrix.config_hashes["wide"]
    right = rows[rows["model"] == "right"].sort_values("origin_index")["crps"].to_numpy()
    np.testing.assert_array_equal(matrix.values["right"].to_numpy(), right)


def test_loss_matrix_flags_rows_with_missing_reason() -> None:
    rows = _scored_rows({"right": 1.0, "wide": 3.0}, proxy_nan_at=(200, 250))
    flagged = rows[rows["missing_reason"] != ""]
    assert set(flagged["missing_reason"]) == {"proxy_nan"}
    assert flagged["crps"].notna().all()  # CRPS itself was scorable on those rows
    default = loss_matrix(rows, "crps")
    assert default.n_flagged == {"right": 2, "wide": 2}
    assert default.values.isna().sum().to_dict() == {"right": 2, "wide": 2}
    by_score = loss_matrix(rows, "crps", policy="score")
    assert by_score.n_flagged == {"right": 0, "wide": 0}
    qlike = loss_matrix(rows, "qlike", policy="score")
    assert qlike.n_flagged == {"right": 2, "wide": 2}


def test_loss_matrix_checks_cells_share_a_series_when_given_a_store(tmp_path: Path) -> None:
    """origin_index is positional: with a store, cells on different data are refused."""
    store = ResultsStore(tmp_path)
    rng = np.random.default_rng(3)
    series = rng.normal(0.0, SIGMA, size=300)
    proxy = series**2 + 1e-8
    splitter = RollingOriginSplitter(window=100, horizon=1, step=1, refit_every=50)

    def cell(
        label: str, scale: float, data: NDArray[np.float64], prox: NDArray[np.float64]
    ) -> pd.DataFrame:
        return run_backtest(
            lambda: Oracle(SIGMA * scale, label),
            data,
            prox,
            splitter,
            seed=1,
            asset="SIM",
            proxy_name="r2",
            store=store,
        )

    same = pd.concat([cell("right", 1.0, series, proxy), cell("wide", 3.0, series, proxy)])
    checked = loss_matrix(same, "crps", store=store)
    assert checked.models == ("right", "wide")
    shifted = pd.concat(
        [cell("right", 1.0, series, proxy), cell("late", 1.0, series[1:], proxy[1:])]
    )
    with pytest.raises(ValueError, match="different data"):
        loss_matrix(shifted, "crps", store=store)
    loss_matrix(shifted, "crps")  # without a store the alignment is the caller's claim
    other_proxy = pd.concat(
        [cell("right", 1.0, series, proxy), cell("parkinson-ish", 1.0, series, proxy * 0.9)]
    )
    loss_matrix(other_proxy, "crps", store=store)  # CRPS never sees the proxy
    with pytest.raises(ValueError, match="'proxy' differs"):
        loss_matrix(other_proxy, "qlike", store=store)


def test_loss_matrix_refuses_ambiguous_inputs() -> None:
    rows = _scored_rows({"right": 1.0, "wide": 3.0})
    with pytest.raises(ValueError, match="one asset"):
        loss_matrix(pd.concat([rows, rows.assign(asset="OTHER")]), "crps")
    with pytest.raises(ValueError, match="one horizon"):
        loss_matrix(pd.concat([rows, rows.assign(horizon=2)]), "crps")
    with pytest.raises(ValueError, match="more than one config_hash"):
        loss_matrix(pd.concat([rows, rows.assign(config_hash="0" * 64)]), "crps")
    with pytest.raises(ValueError, match="duplicate"):
        loss_matrix(pd.concat([rows, rows]), "crps")
    with pytest.raises(ValueError, match="missing columns"):
        loss_matrix(rows.drop(columns=["missing_reason"]), "crps")
    with pytest.raises(ValueError, match="empty"):
        loss_matrix(rows.iloc[:0], "crps")


def test_compare_models_end_to_end_on_scored_rows() -> None:
    """A right-sized oracle against two mis-scaled ones: MCS keeps the right one."""
    rows = _scored_rows({"right": 1.0, "wide": 3.0, "narrow": 0.4})
    matrix = loss_matrix(rows, "log_score")
    comparison = compare_models(matrix, seed=11, n_boot=1000)
    assert comparison.mcs.included == ("right",)
    assert set(comparison.mcs.excluded) == {"wide", "narrow"}
    assert comparison.mcs.horizon == 1 and comparison.dm.horizon == 1 and comparison.dm.lag == 0
    assert comparison.dm.p_value.loc["right", "wide"] < 1e-3
    assert comparison.dm.statistic.loc["right", "wide"] < 0  # right lost less
    table = comparison.table()
    assert (
        list(table["model"]) == ["right", "narrow", "wide"]
        or table["mean_loss"].is_monotonic_increasing
    )
    assert table.loc[table["model"] == "right", "in_mcs"].item()
    assert (table["n"] == comparison.mcs.n).all() and (table["n_dropped"] == 0).all()
    assert len(comparison.config_hash) == 64


def test_compare_models_reports_dropped_origins_everywhere() -> None:
    rows = _scored_rows({"right": 1.0, "wide": 3.0}, proxy_nan_at=(200,))
    comparison = compare_models(loss_matrix(rows, "qlike"), seed=1, n_boot=200)
    assert comparison.mcs.n_dropped == 1
    assert int(comparison.dm.n_dropped.loc["right", "wide"]) == 1
    assert comparison.mcs.n == int(comparison.dm.n.loc["right", "wide"])
