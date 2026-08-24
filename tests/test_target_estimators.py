"""Decisive validation of the M2 scoring target, against a KNOWN true variance.

M1 report §4.4. HAR forecasts — and is scored on — the variance of the
close-to-close return. A range proxy (Parkinson, Garman-Klass, Rogers-Satchell)
estimates only the *intraday* open-to-close variance, so scoring a
close-to-close forecast against it is biased low by the overnight share, for
every model, independent of model error.

``make_toy_asset`` simulates each day's variance as two independent pieces —
an overnight jump and an open-to-close random walk — whose variances sum to
the day's ``true_variance``, which the generator records. That lets this file
measure each estimator's bias and sampling variance against the *truth*, not
against another estimator. The claims the decision rests on:

1. ``overnight_plus_range_variance`` targets the close-to-close variance: its
   bias is far below Parkinson's, which structurally misses the overnight gap.
2. It is far less noisy than the squared daily return, the other unbiased
   close-to-close estimator.

Numbers are asymptotic (20k days), so they test the estimators, not a
particular 700-day draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from volbench.benchmarks.make_toy_asset import OVERNIGHT_SHARE, simulate_ohlc
from volbench.data import (
    garman_klass,
    overnight_plus_range_variance,
    parkinson,
    squared_return,
)


@pytest.fixture(scope="module")
def panel() -> dict[str, np.ndarray]:
    """20k days at fine resolution: discretization negligible, so what remains
    is each estimator's structural bias against the true close-to-close var."""
    frame = simulate_ohlc(n_days=20000, seed=2026, intraday_steps=5000)
    o, h, low, c = (frame[k] for k in ("open", "high", "low", "close"))
    truth = frame["true_variance"].to_numpy()[1:]  # drop first: no C_{-1}
    return {
        "truth": truth,
        "overnight_plus_range": overnight_plus_range_variance(o, h, low, c).to_numpy()[1:],
        "parkinson": parkinson(h, low).to_numpy()[1:],
        "garman_klass": garman_klass(o, h, low, c).to_numpy()[1:],
        "squared_return": squared_return(c).to_numpy()[1:],
    }


def _relative_bias(est: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(est) / np.mean(truth) - 1.0)


def _sampling_variance(est: np.ndarray, truth: np.ndarray) -> float:
    """Variance of the per-day estimator/truth ratio: dispersion around the target."""
    return float(np.var(est / truth))


class TestTargetsCloseToCloseVariance:
    def test_overnight_plus_range_is_close_to_unbiased(self, panel: dict[str, np.ndarray]) -> None:
        bias = _relative_bias(panel["overnight_plus_range"], panel["truth"])
        assert abs(bias) < 0.04, bias  # ~3% asymptotic (residual RS discretization); <1% on
        # the committed 700-day draw. Either way a fraction of Parkinson's structural miss.

    def test_parkinson_misses_the_overnight_gap(self, panel: dict[str, np.ndarray]) -> None:
        """Structural, not noise: Parkinson sees only the open-to-close path,
        so it recovers ~(1 - OVERNIGHT_SHARE) of the close-to-close variance."""
        bias = _relative_bias(panel["parkinson"], panel["truth"])
        assert bias < -0.06  # materially low
        assert bias == pytest.approx(-OVERNIGHT_SHARE, abs=0.03)  # and by ~the overnight share

    def test_new_target_bias_is_far_below_parkinson_and_garman_klass(
        self, panel: dict[str, np.ndarray]
    ) -> None:
        new = abs(_relative_bias(panel["overnight_plus_range"], panel["truth"]))
        park = abs(_relative_bias(panel["parkinson"], panel["truth"]))
        gk = abs(_relative_bias(panel["garman_klass"], panel["truth"]))
        assert new < 0.35 * park  # about a third of Parkinson's bias or less
        assert new < 0.35 * gk


class TestFarLessNoisyThanSquaredReturn:
    def test_squared_return_is_unbiased_but_noisy(self, panel: dict[str, np.ndarray]) -> None:
        # The squared daily return is the other unbiased close-to-close proxy,
        # so its low bias is not the issue — its dispersion is.
        assert abs(_relative_bias(panel["squared_return"], panel["truth"])) < 0.03

    def test_new_target_sampling_variance_is_well_below_squared_return(
        self, panel: dict[str, np.ndarray]
    ) -> None:
        new = _sampling_variance(panel["overnight_plus_range"], panel["truth"])
        sq = _sampling_variance(panel["squared_return"], panel["truth"])
        assert new < 0.5 * sq  # ~0.28 vs ~1.9 on the committed fixture: ~7x tighter


class TestComponentsAddUpInTheGenerator:
    """The validation only means something if the fixture's truth really is the
    close-to-close variance and the pieces are independent as claimed."""

    def test_true_variance_equals_realized_close_to_close_variance(self) -> None:
        frame = simulate_ohlc(n_days=40000, seed=5, intraday_steps=64)
        close = frame["close"].to_numpy()
        overnight_open = frame["open"].to_numpy()[1:]
        r = np.log(close[1:] / close[:-1])  # close-to-close log return
        # Zero-mean by construction, so E[r^2] is the variance; it must match
        # the mean of the recorded true_variance.
        realized = float(np.mean(r**2))
        recorded = float(np.mean(frame["true_variance"].to_numpy()[1:]))
        assert realized == pytest.approx(recorded, rel=0.03)
        # And the overnight piece carries ~OVERNIGHT_SHARE of it.
        overnight = np.log(overnight_open / close[:-1])
        assert float(np.mean(overnight**2)) / float(np.mean(r**2)) == pytest.approx(
            OVERNIGHT_SHARE, abs=0.02
        )
