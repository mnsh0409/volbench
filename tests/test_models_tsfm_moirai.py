"""Moirai-2 adapter — same three layers as the Chronos tests.

- ``TestMocked``: the adapter class with a fake backend (CI).
- ``TestBackendGlue``: ``MoiraiBackend`` against a stub forecaster; needs
  torch (``inference_mode``) but no weights.
- ``TestRealCheckpoint`` (``@pytest.mark.tsfm``): ``Salesforce/moirai-2.0-R-small``,
  including the scale test that justifies ``input_scale``'s default.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from tsfm_fakes import FakeBackend, realized_variance

from volbench.models import Moirai
from volbench.models.tsfm_common import resolve_hf_revision, tail_closed_grid_mean
from volbench.models.tsfm_moirai import DEFAULT_MOIRAI_CHECKPOINT, MoiraiBackend

_SHA = re.compile(r"^[0-9a-f]{40}$")
_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class TestMocked:
    def test_name_and_defaults(self) -> None:
        model = Moirai(backend=FakeBackend())
        assert model.name == "moirai_2_0_r_small"
        assert model.checkpoint == DEFAULT_MOIRAI_CHECKPOINT

    def test_fit_predict_update_with_a_fake_backend(self) -> None:
        rv = realized_variance()
        fitted = Moirai(backend=FakeBackend(), context_length=256).fit(rv)
        dist = fitted.predict(1)
        assert 0.0 < dist.variance() < 1e-2
        assert np.array_equal(fitted.update(rv).context, fitted.context)

    def test_device_is_passed_to_the_loader_but_kept_out_of_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import volbench.models.tsfm_moirai as module

        seen: list[tuple[Any, ...]] = []

        def loader(*args: Any) -> Any:
            seen.append(args)
            return FakeBackend()

        monkeypatch.setattr(module, "_load_moirai", loader)
        assert Moirai(device="cpu").spec() == Moirai(device="cuda").spec()
        assert seen == [
            (DEFAULT_MOIRAI_CHECKPOINT, None, "cpu"),
            (DEFAULT_MOIRAI_CHECKPOINT, None, "cuda"),
        ]


class _Module:
    quantile_levels = _LEVELS
    patch_size = 16
    max_seq_len = 512


class _StubForecaster:
    """``hparams_context`` + ``predict`` + ``module``: the surface the backend uses."""

    def __init__(self) -> None:
        self.module = _Module()
        self.hparams_seen: list[tuple[int, int]] = []
        self.calls: list[np.ndarray] = []

    @contextmanager
    def hparams_context(self, prediction_length: int, context_length: int) -> Iterator[Any]:
        self.hparams_seen.append((prediction_length, context_length))
        yield self

    def predict(self, past_target: list[np.ndarray]) -> np.ndarray:
        ctx = np.asarray(past_target[0])
        self.calls.append(ctx.copy())
        h, _ = self.hparams_seen[-1]
        level = float(ctx[-22:].mean())
        factors = np.array([0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.7, 2.2])
        steps = 1.0 + 0.01 * np.arange(1, h + 1)
        return (level * factors[:, None] * steps[None, :])[None, :, :]  # (1, q, h)


class TestBackendGlue:
    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    def test_context_goes_in_as_float32_and_quantiles_come_back_as_h_by_q(self) -> None:
        stub = _StubForecaster()
        backend = MoiraiBackend(stub, checkpoint="x/y", revision="f" * 40)
        ctx = realized_variance(100) * 1e4
        out = backend.forecast(ctx, 3)
        assert stub.hparams_seen == [(3, 100)]
        (sent,) = stub.calls
        assert sent.dtype == np.float32 and np.allclose(sent, ctx.astype(np.float32))
        assert out.taus == _LEVELS
        assert out.values.shape == (3, 9) and out.native_mean is None
        assert np.all(np.diff(out.values, axis=1) > 0)
        assert np.allclose(out.values[2] / out.values[0], 1.03 / 1.01)

    def test_identity_and_limits(self) -> None:
        backend = MoiraiBackend(
            _StubForecaster(), checkpoint="x/y", revision="f" * 40, versions={"uni2ts": "v"}
        )
        assert backend.max_context == (512 - 8) * 16
        ident = backend.identity()
        assert ident["backend"] == "moirai" and ident["patch_size"] == 16
        assert ident["uni2ts"] == "v"
        with pytest.raises(ValueError, match="exceeds"):
            backend.forecast(realized_variance(100), 8 * 16 + 1)


@pytest.mark.tsfm
class TestRealCheckpoint:
    def test_spec_pins_the_weights(self) -> None:
        spec = Moirai().spec()
        assert _SHA.match(spec["revision"])
        assert spec["revision"] == resolve_hf_revision(DEFAULT_MOIRAI_CHECKPOINT)
        assert {"uni2ts", "torch", "patch_size"} <= set(spec)

    def test_bit_identical_twice(self) -> None:
        rv = realized_variance(1000)
        a, b = Moirai().fit(rv), Moirai().fit(rv)
        assert np.array_equal(a.rv_forecast(5).values, b.rv_forecast(5).values)
        assert a.predict(1) == b.predict(1)
        assert a.predict(5) == b.predict(5)

    def test_forecast_is_a_sane_daily_variance(self) -> None:
        rv = realized_variance(1000)
        fitted = Moirai().fit(rv)
        dist = fitted.predict(1)
        meta = fitted.spec()["rv_forecasts"]["1"]
        grid = np.asarray(meta["values"])
        assert meta["clipped_at_zero"] == 0
        # sigma = sqrt(vhat); squaring it back is exact only to an ulp
        assert dist.variance() == pytest.approx(
            tail_closed_grid_mean(np.asarray(meta["taus"]), grid), rel=1e-12
        )
        # The closure only re-expresses the two atoms, so the scored mean must
        # still sit strictly above the flat-tailed one and inside the grid's
        # own outer levels.
        assert meta["tail_closure"] == "lognormal"
        assert meta["flat_tail_mean"] < dist.variance()
        assert grid[0] <= dist.variance() <= grid[-1]
        level = float(np.mean(rv[-22:]))
        assert 0.3 * level < dist.variance() < 3.0 * level

    def test_the_scaler_epsilon_is_why_input_scale_defaults_to_1e4(self) -> None:
        """At raw daily-variance units Moirai's ``sqrt(var + 1e-5)`` flattens the
        series: the grid collapses onto the level. In percent-squared and
        beyond it is stable. This is the evidence behind the default."""
        rv = realized_variance(1000)

        def spread(scale: float) -> float:
            fitted = Moirai(input_scale=scale).fit(rv)
            fitted.predict(1)
            grid = np.asarray(fitted.spec()["rv_forecasts"]["1"]["values"])
            return float((grid[-1] - grid[0]) / grid[4])

        assert spread(1.0) < 0.05
        assert spread(1e4) > 0.5
        v4 = Moirai(input_scale=1e4).fit(rv).predict(1).variance()
        v6 = Moirai(input_scale=1e6).fit(rv).predict(1).variance()
        assert v6 == pytest.approx(v4, rel=1e-3)

    def test_update_moves_with_the_context(self) -> None:
        rv = realized_variance(1100)
        fitted = Moirai(context_length=512).fit(rv[:1000])
        assert fitted.update(rv[:1000]).predict(1) == fitted.predict(1)
        shocked = rv[1:1001].copy()
        shocked[-1] *= 10.0
        assert fitted.update(shocked).predict(1).variance() > fitted.predict(1).variance()

    def test_first_step_does_not_depend_on_the_requested_horizon(self) -> None:
        fitted = Moirai().fit(realized_variance(1000))
        assert np.array_equal(fitted.rv_forecast(1).values[0], fitted.rv_forecast(5).values[0])
