"""TimeGPT adapter — the gates first, then the request/response glue, then the API.

- ``TestGates`` / ``TestMocked`` / ``TestBackendGlue`` run in CI: no key, no
  network; the stub client records the request the adapter would send.
- ``TestRealApi`` carries ``@pytest.mark.timegpt``: skipped unless
  ``NIXTLA_API_KEY`` is set, always skipped under ``CI`` (tests/conftest.py).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tsfm_fakes import FakeBackend, realized_variance

from volbench.models import TimeGPT
from volbench.models.tsfm_timegpt import (
    DEFAULT_TIMEGPT_MODEL,
    TIMEGPT_KEY_ENV,
    TimeGPTBackend,
)

_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class TestGates:
    def test_disabled_by_default_and_refuses_before_reading_any_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TIMEGPT_KEY_ENV, "would-be-a-real-key")
        monkeypatch.delitem(sys.modules, "nixtla", raising=False)
        model = TimeGPT()
        assert model.enabled is False
        with pytest.raises(RuntimeError, match="opt-in"):
            model.fit(realized_variance())
        assert "nixtla" not in sys.modules  # nothing was even imported

    def test_enabled_without_a_key_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(TIMEGPT_KEY_ENV, raising=False)
        with pytest.raises(RuntimeError, match=TIMEGPT_KEY_ENV):
            TimeGPT(enabled=True).fit(realized_variance())
        monkeypatch.setenv(TIMEGPT_KEY_ENV, "")
        with pytest.raises(RuntimeError, match=TIMEGPT_KEY_ENV):
            TimeGPT(enabled=True).fit(realized_variance())

    def test_spec_needs_neither_key_nor_network_and_never_contains_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TIMEGPT_KEY_ENV, "sekrit-value")
        spec = TimeGPT(enabled=True).spec()
        text = json.dumps(spec, sort_keys=True)
        assert "sekrit" not in text
        assert spec["model"] == "timegpt_1"
        assert spec["checkpoint"] == DEFAULT_TIMEGPT_MODEL
        assert spec["revision"] == "hosted-api-unpinned"
        assert spec["quantile_levels"] == list(_LEVELS)
        assert "nixtla" in spec
        assert spec == TimeGPT(enabled=False).spec()  # the flag is not a hyperparameter

    def test_quantiles_are_validated(self) -> None:
        with pytest.raises(ValueError, match="quantiles"):
            TimeGPT(quantiles=(0.5,))
        with pytest.raises(ValueError, match="quantiles"):
            TimeGPT(quantiles=(0.0, 0.5))


class TestMocked:
    def test_fit_predict_update_with_a_fake_backend(self) -> None:
        rv = realized_variance()
        model = TimeGPT(backend=FakeBackend(native_mean=True), context_length=256)
        fitted = model.fit(rv)  # an injected backend needs no opt-in: nothing leaves the box
        dist = fitted.predict(1)
        assert 0.0 < dist.variance() < 1e-2
        assert fitted.spec()["rv_forecasts"]["1"]["native_mean"] > 0.0
        assert np.array_equal(fitted.update(rv).context, fitted.context)
        assert model.spec()["revision"] == "f" * 40  # the injected identity wins

    def test_name_follows_the_api_model(self) -> None:
        assert TimeGPT(backend=FakeBackend()).name == "timegpt_1"
        assert TimeGPT(backend=FakeBackend(), api_model="timegpt-1-long-horizon").name == (
            "timegpt_1_long_horizon"
        )


class _StubClient:
    def __init__(self, *, shuffle: bool = False, drop_point: bool = False, bad_ds: bool = False):
        self.calls: list[dict[str, Any]] = []
        self.shuffle, self.drop_point, self.bad_ds = shuffle, drop_point, bad_ds

    def forecast(
        self, df: pd.DataFrame, h: int, freq: Any, quantiles: list[float], model: str
    ) -> pd.DataFrame:
        self.calls.append(
            {"df": df.copy(), "h": h, "freq": freq, "quantiles": quantiles, "model": model}
        )
        n = len(df)
        level = float(df["y"].to_numpy()[-22:].mean())
        ds = np.arange(n, n + h) + (1 if self.bad_ds else 0)
        out = pd.DataFrame({"unique_id": "rv", "ds": ds})
        if not self.drop_point:
            out["TimeGPT"] = level * 1.05
        for q in quantiles:
            step = 1.0 + 0.01 * (ds - n + 1)
            out[f"TimeGPT-q-{round(q * 100)}"] = level * (0.5 + q) * step
        return out.iloc[::-1] if self.shuffle else out


class TestBackendGlue:
    def test_request_is_the_context_on_an_integer_calendar(self) -> None:
        client = _StubClient()
        backend = TimeGPTBackend(client, api_model="timegpt-1", quantiles=(0.9, 0.1, 0.5))
        ctx = realized_variance(300) * 1e4
        out = backend.forecast(ctx, 2)
        (call,) = client.calls
        df = call["df"]
        assert list(df.columns) == ["unique_id", "ds", "y"]
        assert df["ds"].dtype == np.int64 and df["ds"].tolist() == list(range(300))
        assert np.array_equal(df["y"].to_numpy(), ctx)
        assert call["h"] == 2 and call["freq"] == 1 and call["model"] == "timegpt-1"
        assert call["quantiles"] == [0.1, 0.5, 0.9]  # sorted, as the grid needs them
        assert out.taus == (0.1, 0.5, 0.9)
        assert out.values.shape == (2, 3)
        assert out.native_mean is not None and out.native_mean.shape == (2,)
        level = ctx[-22:].mean()
        assert out.values[1, 2] == pytest.approx(level * 1.4 * 1.02)

    def test_response_rows_are_ordered_by_ds_before_use(self) -> None:
        backend = TimeGPTBackend(_StubClient(shuffle=True), api_model="m", quantiles=_LEVELS)
        out = backend.forecast(realized_variance(100), 3)
        assert np.all(out.values[1:] > out.values[:-1])  # step factor increases with ds

    def test_a_response_that_is_not_the_next_h_steps_is_refused(self) -> None:
        backend = TimeGPTBackend(_StubClient(bad_ds=True), api_model="m", quantiles=_LEVELS)
        with pytest.raises(RuntimeError, match="after the context end"):
            backend.forecast(realized_variance(100), 1)

    def test_missing_point_column_means_no_native_mean(self) -> None:
        backend = TimeGPTBackend(_StubClient(drop_point=True), api_model="m", quantiles=_LEVELS)
        assert backend.forecast(realized_variance(100), 1).native_mean is None

    def test_identity(self) -> None:
        backend = TimeGPTBackend(
            _StubClient(), api_model="timegpt-1", quantiles=_LEVELS, versions={"nixtla": "v"}
        )
        ident = backend.identity()
        assert ident == {
            "backend": "timegpt",
            "checkpoint": "timegpt-1",
            "revision": "hosted-api-unpinned",
            "quantile_levels": list(_LEVELS),
            "nixtla": "v",
        }
        assert backend.max_context >= 10_000


@pytest.mark.timegpt
class TestRealApi:
    """Only with a key, never under CI. Reports rather than assumes determinism."""

    def test_forecast_and_repeatability(self) -> None:
        rv = realized_variance(500)
        model = TimeGPT(enabled=True, context_length=256)
        a = model.fit(rv)
        dist = a.predict(1)
        meta = a.spec()["rv_forecasts"]["1"]
        grid = np.asarray(meta["values"])
        assert grid[0] <= dist.variance() <= grid[-1]
        level = float(np.mean(rv[-22:]))
        assert 0.3 * level < dist.variance() < 3.0 * level
        b = model.fit(rv)
        assert np.array_equal(a.rv_forecast(1).values, b.rv_forecast(1).values), (
            "TimeGPT returned different forecasts for the same request — "
            "the hosted model is not reproducible; keep it out of the headline"
        )
