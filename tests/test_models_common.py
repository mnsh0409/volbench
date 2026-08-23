"""Invariants every return-based model must satisfy: valid Distribution out,
JSON-serializable spec(), spec() stable/differing appropriately.

HAR is excluded from the shared fixtures here since its `fit()` contract
takes a realized-variance series, not returns (see test_models_har.py).
"""

import json

import numpy as np
import pytest

from volbench.models import EWMA, GARCH, NaiveVol, gjr_garch

RETURN_MODELS = [
    NaiveVol(),
    EWMA(),
    GARCH(dist="normal"),
    GARCH(dist="studentst"),
    gjr_garch(dist="normal"),
    gjr_garch(dist="studentst"),
]


def _returns(seed: int, n: int = 500) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, n)


@pytest.mark.parametrize("model", RETURN_MODELS, ids=lambda m: m.name)
def test_predict_returns_distribution_with_positive_variance(model) -> None:
    fitted = model.fit(_returns(seed=1))
    dist = fitted.predict(1)
    samples = dist.sample(5000, seed=0)
    assert np.isfinite(samples).all()
    assert np.var(samples) > 0.0


@pytest.mark.parametrize("model", RETURN_MODELS, ids=lambda m: m.name)
def test_spec_is_json_serializable(model) -> None:
    json.dumps(model.spec(), sort_keys=True)


@pytest.mark.parametrize("model", RETURN_MODELS, ids=lambda m: m.name)
def test_fitted_spec_is_json_serializable(model) -> None:
    fitted = model.fit(_returns(seed=2))
    json.dumps(fitted.spec(), sort_keys=True)


def test_spec_stable_across_identical_constructions() -> None:
    assert NaiveVol().spec() == NaiveVol().spec()
    assert EWMA(lambda_=0.9).spec() == EWMA(lambda_=0.9).spec()
    assert GARCH(dist="normal").spec() == GARCH(dist="normal").spec()
    assert gjr_garch(dist="studentst").spec() == gjr_garch(dist="studentst").spec()


def test_spec_differs_across_different_models_and_hyperparameters() -> None:
    specs = [m.spec() for m in RETURN_MODELS]
    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            assert a != b
