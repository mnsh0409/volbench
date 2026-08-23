"""ForecastModel/FittedModel protocol conformance."""

import numpy as np

from volbench.dist import Distribution
from volbench.models import EWMA, GARCH, HAR, NaiveVol
from volbench.models.base import FittedModel, ForecastModel


class TestReturnBasedModelsConformProtocol:
    def test_naive_conforms(self) -> None:
        model = NaiveVol()
        assert isinstance(model, ForecastModel)
        fitted = model.fit(np.array([0.01, -0.02, 0.015, -0.005, 0.02]))
        assert isinstance(fitted, FittedModel)
        assert isinstance(fitted.predict(1), Distribution)

    def test_ewma_conforms(self) -> None:
        model = EWMA()
        assert isinstance(model, ForecastModel)
        fitted = model.fit(np.array([0.01, -0.02, 0.015, -0.005, 0.02]))
        assert isinstance(fitted, FittedModel)

    def test_garch_conforms(self) -> None:
        model = GARCH()
        assert isinstance(model, ForecastModel)
        rng = np.random.default_rng(0)
        fitted = model.fit(rng.normal(0.0, 0.01, 200))
        assert isinstance(fitted, FittedModel)

    def test_har_conforms(self) -> None:
        model = HAR()
        assert isinstance(model, ForecastModel)
        rng = np.random.default_rng(0)
        rv = np.abs(rng.normal(0.0, 1e-4, 60)) + 1e-6
        fitted = model.fit(rv)
        assert isinstance(fitted, FittedModel)
