"""Proxy-robust point losses."""

import pytest

from volbench import mse, pinball, qlike


def test_qlike_zero_at_perfect_forecast() -> None:
    assert qlike(2.5, 2.5) == pytest.approx(0.0)


def test_qlike_asymmetry_penalizes_underprediction() -> None:
    # Under-predicting variance by 2x costs more than over-predicting by 2x.
    assert qlike(1.0, 2.0) > qlike(4.0, 2.0) > 0.0


def test_qlike_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        qlike(0.0, 1.0)
    with pytest.raises(ValueError):
        qlike(1.0, -1.0)


def test_mse_symmetric() -> None:
    assert mse(1.0, 3.0) == mse(3.0, 1.0) == pytest.approx(4.0)


def test_pinball_known_values() -> None:
    # y above q: tau * (y - q); y below q: (1 - tau) * (q - y)
    assert pinball(y=2.0, q=1.0, tau=0.05) == pytest.approx(0.05)
    assert pinball(y=0.0, q=1.0, tau=0.05) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        pinball(0.0, 0.0, tau=1.0)
