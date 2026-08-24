"""TimesFM adapter — same three layers as the Chronos tests.

- ``TestMocked``: the adapter class with a fake backend (CI).
- ``TestBackendGlue``: ``TimesFMBackend`` against a stub model; needs the
  ``timesfm`` package for ``ForecastConfig`` but no weights.
- ``TestRealCheckpoint`` (``@pytest.mark.tsfm``): ``google/timesfm-2.5-200m-pytorch``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
from tsfm_fakes import FakeBackend, realized_variance

from volbench.models import TimesFM, TimesFMForecastOptions
from volbench.models.tsfm_common import quantile_grid_mean, resolve_hf_revision
from volbench.models.tsfm_timesfm import DEFAULT_TIMESFM_CHECKPOINT, TimesFMBackend

_SHA = re.compile(r"^[0-9a-f]{40}$")
_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TestMocked:
    def test_name_and_defaults(self) -> None:
        model = TimesFM(backend=FakeBackend())
        assert model.name == "timesfm_2_5_200m_pytorch"
        assert model.checkpoint == DEFAULT_TIMESFM_CHECKPOINT
        assert model.options == TimesFMForecastOptions()
        assert model.options.max_horizon == 128

    def test_fit_predict_update_with_a_fake_backend(self) -> None:
        rv = realized_variance()
        fitted = TimesFM(backend=FakeBackend(native_mean=True), context_length=256).fit(rv)
        dist = fitted.predict(1)
        assert 0.0 < dist.variance() < 1e-2
        assert fitted.spec()["rv_forecasts"]["1"]["native_mean"] > 0.0
        assert np.array_equal(fitted.update(rv).context, fitted.context)

    def test_options_are_part_of_the_hashed_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fake backend cannot see the options, so this pins where they enter."""
        import volbench.models.tsfm_timesfm as module

        seen: list[Any] = []

        def loader(checkpoint: str, revision: str | None, options: Any) -> Any:
            seen.append((checkpoint, revision, options))
            return FakeBackend(identity={"backend": "timesfm", **options.as_dict()})

        monkeypatch.setattr(module, "_load_timesfm", loader)
        default = TimesFM().spec()
        tuned = TimesFM(options=TimesFMForecastOptions(normalize_inputs=False)).spec()
        assert default != tuned
        assert default["normalize_inputs"] is True and tuned["normalize_inputs"] is False
        assert seen[0][2] == TimesFMForecastOptions()


class _Definition:
    quantiles = _LEVELS
    context_limit = 16384
    input_patch_len = 32


class _Inner:
    config = _Definition()


class _StubTimesFM:
    """``compile`` + ``forecast`` + ``model.config``: the surface the backend uses."""

    def __init__(self) -> None:
        self.model = _Inner()
        self.compiled: list[Any] = []
        self.calls: list[tuple[np.ndarray, int]] = []

    def compile(self, config: Any) -> None:
        self.compiled.append(config)

    def forecast(self, horizon: int, inputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        ctx = np.asarray(inputs[0])
        self.calls.append((ctx.copy(), horizon))
        level = float(ctx[-22:].mean())
        factors = np.array([1.1, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.7, 2.2])  # point head first
        steps = 1.0 + 0.01 * np.arange(1, horizon + 1)
        q = level * factors[None, :] * steps[:, None]
        return q[None, :, 5], q[None, :, :]


class TestBackendGlue:
    @pytest.fixture(autouse=True)
    def _timesfm(self) -> None:
        pytest.importorskip("timesfm")

    def _backend(self, **options: Any) -> tuple[_StubTimesFM, TimesFMBackend]:
        stub = _StubTimesFM()
        backend = TimesFMBackend(
            stub,
            checkpoint="x/y",
            revision="f" * 40,
            options=TimesFMForecastOptions(**options),
            versions={"timesfm": "v"},
        )
        return stub, backend

    def test_point_head_is_native_mean_and_the_rest_are_the_grid(self) -> None:
        stub, backend = self._backend()
        ctx = realized_variance(100) * 1e4
        out = backend.forecast(ctx, 2)
        assert out.taus == tuple(_LEVELS)
        assert out.values.shape == (2, 9)
        assert out.native_mean is not None and out.native_mean.shape == (2,)
        level = ctx[-22:].mean()
        assert out.native_mean[0] == pytest.approx(level * 1.1 * 1.01)
        assert out.values[0, 4] == pytest.approx(level * 1.0 * 1.01)
        ((sent, h),) = stub.calls
        assert h == 2 and np.array_equal(sent, ctx)

    def test_compiles_to_the_patch_boundary_and_only_on_change(self) -> None:
        stub, backend = self._backend(max_horizon=64, normalize_inputs=False)
        backend.forecast(realized_variance(100), 1)
        backend.forecast(realized_variance(100), 3)
        backend.forecast(realized_variance(128), 1)
        backend.forecast(realized_variance(129), 1)
        assert [c.max_context for c in stub.compiled] == [128, 160]
        assert all(c.max_horizon == 64 for c in stub.compiled)
        assert all(c.normalize_inputs is False for c in stub.compiled)
        assert all(c.use_continuous_quantile_head is True for c in stub.compiled)

    def test_horizon_beyond_the_compiled_one_is_refused(self) -> None:
        _, backend = self._backend(max_horizon=64)
        with pytest.raises(ValueError, match="max_horizon"):
            backend.forecast(realized_variance(100), 65)

    def test_identity_and_limits(self) -> None:
        _, backend = self._backend(max_horizon=128)
        assert backend.max_context == 16384 - 128
        ident = backend.identity()
        assert ident["backend"] == "timesfm"
        assert ident["torch_compile"] is False
        assert ident["fix_quantile_crossing"] is True
        assert ident["timesfm"] == "v"


@pytest.mark.tsfm
class TestRealCheckpoint:
    def test_spec_pins_the_weights(self) -> None:
        spec = TimesFM().spec()
        assert _SHA.match(spec["revision"])
        assert spec["revision"] == resolve_hf_revision(DEFAULT_TIMESFM_CHECKPOINT)
        assert {"timesfm", "torch", "max_horizon", "normalize_inputs"} <= set(spec)

    def test_bit_identical_twice(self) -> None:
        rv = realized_variance(1000)
        a, b = TimesFM().fit(rv), TimesFM().fit(rv)
        assert np.array_equal(a.rv_forecast(5).values, b.rv_forecast(5).values)
        assert a.predict(1) == b.predict(1)
        assert a.predict(5) == b.predict(5)

    def test_forecast_is_a_sane_daily_variance_with_a_positive_point_head(self) -> None:
        rv = realized_variance(1000)
        fitted = TimesFM().fit(rv)
        dist = fitted.predict(1)
        meta = fitted.spec()["rv_forecasts"]["1"]
        grid = np.asarray(meta["values"])
        assert meta["crossings_rearranged"] == 0  # fix_quantile_crossing did its job upstream
        assert meta["clipped_at_zero"] == 0  # infer_is_positive
        assert meta["native_mean"] > 0.0
        assert grid[0] <= dist.variance() <= grid[-1]
        # sigma = sqrt(vhat); squaring it back is exact only to an ulp
        assert dist.variance() == pytest.approx(
            quantile_grid_mean(np.asarray(meta["taus"]), grid), rel=1e-12
        )
        level = float(np.mean(rv[-22:]))
        assert 0.3 * level < dist.variance() < 3.0 * level

    def test_scale_stable_at_the_default_units(self) -> None:
        rv = realized_variance(1000)
        v4 = TimesFM(input_scale=1e4).fit(rv).predict(1).variance()
        v6 = TimesFM(input_scale=1e6).fit(rv).predict(1).variance()
        assert v6 == pytest.approx(v4, rel=1e-3)

    def test_update_moves_with_the_context(self) -> None:
        rv = realized_variance(1100)
        fitted = TimesFM(context_length=512).fit(rv[:1000])
        assert fitted.update(rv[:1000]).predict(1) == fitted.predict(1)
        shocked = rv[1:1001].copy()
        shocked[-1] *= 10.0
        assert fitted.update(shocked).predict(1).variance() > fitted.predict(1).variance()

    def test_first_step_does_not_depend_on_the_requested_horizon(self) -> None:
        fitted = TimesFM().fit(realized_variance(1000))
        assert np.array_equal(fitted.rv_forecast(1).values[0], fitted.rv_forecast(5).values[0])
