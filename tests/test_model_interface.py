"""The model interface is ONE definition, and every model actually satisfies it.

Regression test for the M1 integration (docs/phase1_prompts.md stream D,
task 3): during Phase 1, `volbench.evaluate` carried its own local
`ForecastModel`/`FittedModel` Protocols because `volbench.models` was being
built in a parallel stream. Two structural definitions of one interface can
drift apart silently — a model gains a method the evaluator's copy never
learns about, or the evaluator tightens a signature no model implements — and
nothing fails until a backtest does. This file makes that drift a test
failure instead.

The static half is carried by the annotated bindings below: `mypy --strict`
rejects this file if any concrete class stops satisfying the Protocol. The
runtime half is the `isinstance` checks, which catch the same drift for
anyone running only pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

from volbench import evaluate as evaluate_module
from volbench.dist import Distribution
from volbench.models import EWMA, GARCH, HAR, NaiveVol
from volbench.models.base import FittedModel, ForecastModel

# Static conformance: annotating with the Protocol is the assertion. If any
# class drops `name`/`spec()`/`fit()`, or changes a signature, mypy fails here.
UNFITTED: list[ForecastModel] = [NaiveVol(), EWMA(), GARCH(), HAR()]

#: HAR is the exception: it fits on a realized-variance series, not returns
#: (see models/har.py), which is why `_train_for` has to branch at all.


def _returns(n: int = 300, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n)


def _realized_variance(n: int = 300, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(np.log(1e-4), 0.4, size=n))


def _train_for(model: ForecastModel) -> np.ndarray:
    return _realized_variance() if isinstance(model, HAR) else _returns()


class TestOneDefinition:
    def test_evaluate_reexports_the_models_protocols_not_copies_of_them(self) -> None:
        # Identity, not just structural equivalence: `evaluate` must be
        # re-exporting `models.base`'s objects. A second `class ForecastModel
        # (Protocol)` anywhere in the package would fail this.
        assert evaluate_module.ForecastModel is ForecastModel
        assert evaluate_module.FittedModel is FittedModel


class TestUnfittedConformance:
    @pytest.mark.parametrize("model", UNFITTED, ids=lambda m: m.name)
    def test_satisfies_forecast_model_at_runtime(self, model: ForecastModel) -> None:
        assert isinstance(model, ForecastModel)

    @pytest.mark.parametrize("model", UNFITTED, ids=lambda m: m.name)
    def test_spec_is_json_like_and_stable(self, model: ForecastModel) -> None:
        spec = model.spec()
        assert isinstance(spec, dict)
        assert spec == model.spec()
        assert all(isinstance(k, str) for k in spec)


class TestFittedConformance:
    @pytest.mark.parametrize("model", UNFITTED, ids=lambda m: m.name)
    def test_fit_returns_something_satisfying_fitted_model(self, model: ForecastModel) -> None:
        fitted: FittedModel = model.fit(_train_for(model))
        assert isinstance(fitted, FittedModel)
        assert isinstance(fitted.name, str) and fitted.name
        assert isinstance(fitted.spec(), dict)

    @pytest.mark.parametrize("model", UNFITTED, ids=lambda m: m.name)
    def test_predict_returns_a_distribution_over_returns(self, model: ForecastModel) -> None:
        fitted = model.fit(_train_for(model))
        dist = fitted.predict(1)
        assert isinstance(dist, Distribution)
        # Rule 2: predict() is a distribution over the next-period RETURN, so
        # its variance is the variance forecast. Daily units: a daily equity
        # return variance is ~1e-4, nowhere near an annualized ~0.04.
        mean, variance = evaluate_module.forecast_moments(dist)
        assert variance > 0.0
        assert variance < 1.0, "variance looks annualized, not daily"
        assert abs(mean) < 1.0


class TestUpdateCapability:
    """Every baseline re-conditions between refits without re-estimating.

    M1 report §4.3 recorded that no Phase 1 model implemented
    `volbench.evaluate.SupportsUpdate`, so "refit every 21 days" froze each
    forecast for 21 days. Closed on m2/evaluator-hardening: all four implement
    it, and the backtest calls it at every non-refit origin under
    `recondition="daily"`.
    """

    @pytest.mark.parametrize("model", UNFITTED, ids=lambda m: m.name)
    def test_every_model_supports_update(self, model: ForecastModel) -> None:
        fitted = model.fit(_train_for(model))
        assert isinstance(fitted, evaluate_module.SupportsUpdate)
        again: FittedModel = fitted.update(_train_for(model))
        assert isinstance(again, FittedModel)
        assert isinstance(again, evaluate_module.SupportsUpdate)  # and so is its successor

    @pytest.mark.parametrize("model", UNFITTED, ids=lambda m: m.name)
    def test_update_on_the_fit_window_reproduces_the_fit_exactly(
        self, model: ForecastModel
    ) -> None:
        """Re-conditioning on the data a model was just fitted on must change
        nothing — this is the property the refit_every=1 byte-identity
        equivalence rests on, and any drift here is an implementation bug."""
        train = _train_for(model)
        fitted = model.fit(train)
        assert isinstance(fitted, evaluate_module.SupportsUpdate)
        again = fitted.update(train)
        assert evaluate_module.forecast_moments(again.predict(1)) == (
            evaluate_module.forecast_moments(fitted.predict(1))
        )
