"""Chronos adapter.

Three layers, each gated by what it needs:

- ``TestMocked`` — the adapter class with an injected fake backend: no torch,
  no weights, runs in CI.
- ``TestBackendGlue`` — ``ChronosBackend`` against a stub pipeline: needs
  torch (tensors cross the boundary) but no weights; skipped where torch is
  absent.
- ``TestRealCheckpoint`` — ``@pytest.mark.tsfm``: the real Bolt and Chronos-2
  checkpoints on this machine's GPU. Determinism is the gate: same context
  in, same forecast out, twice, bit for bit.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
from tsfm_fakes import FakeBackend, realized_variance

from volbench.models import Chronos
from volbench.models.tsfm_chronos import DEFAULT_CHRONOS_CHECKPOINT, ChronosBackend
from volbench.models.tsfm_common import quantile_grid_mean, resolve_hf_revision

_SHA = re.compile(r"^[0-9a-f]{40}$")
_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TestMocked:
    def test_name_follows_the_checkpoint(self) -> None:
        assert Chronos(backend=FakeBackend()).name == "chronos_bolt_small"
        assert Chronos(backend=FakeBackend(), checkpoint="amazon/chronos-2").name == "chronos_2"
        assert DEFAULT_CHRONOS_CHECKPOINT == "amazon/chronos-bolt-small"

    def test_fit_predict_update_with_a_fake_backend(self) -> None:
        rv = realized_variance()
        fitted = Chronos(backend=FakeBackend(), context_length=128).fit(rv)
        dist = fitted.predict(1)
        assert 0.0 < dist.variance() < 1e-2
        moved = fitted.update(np.append(rv[1:], rv[-1] * 2.0))
        assert moved.predict(1).variance() > dist.variance()

    def test_no_weights_are_loaded_until_needed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import volbench.models.tsfm_chronos as module

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("loader called")

        monkeypatch.setattr(module, "_load_chronos", boom)
        model = Chronos()
        assert model.name == "chronos_bolt_small"  # construction and naming never load
        with pytest.raises(AssertionError, match="loader called"):
            model.spec()  # identity needs the checkpoint's revision
        with pytest.raises(AssertionError, match="loader called"):
            model.fit(realized_variance())

    def test_loader_receives_the_hashed_and_unhashed_knobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import volbench.models.tsfm_chronos as module

        seen: list[tuple[Any, ...]] = []
        fake = FakeBackend()

        def loader(*args: Any) -> Any:
            seen.append(args)
            return fake

        monkeypatch.setattr(module, "_load_chronos", loader)
        model = Chronos(
            checkpoint="amazon/chronos-2", revision="abc", dtype="bfloat16", device="cpu"
        )
        assert model.fit(realized_variance()).backend is fake
        assert seen == [("amazon/chronos-2", "abc", "bfloat16", "cpu")]


class _StubPipeline:
    """The slice of ``BaseChronosPipeline`` the backend touches."""

    quantiles = _LEVELS
    model_context_length = 2048

    def __init__(self, *, as_list: bool) -> None:
        self.as_list = as_list
        self.calls: list[dict[str, Any]] = []

    def predict_quantiles(
        self, inputs: Any, prediction_length: int, quantile_levels: list[float]
    ) -> tuple[Any, Any]:
        import torch

        self.calls.append(
            {"inputs": inputs, "prediction_length": prediction_length, "levels": quantile_levels}
        )
        ctx = inputs[0]
        level = ctx[-22:].mean()
        grid = level * torch.tensor([0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.7, 2.2])
        steps = 1.0 + 0.01 * torch.arange(1, prediction_length + 1, dtype=torch.float32)
        q = grid[None, :] * steps[:, None]  # (h, 9)
        median = q[:, 4]
        if self.as_list:  # Chronos-2 shape: list of (n_variates, h, q)
            return [q[None, :, :]], [median[None, :]]
        return q[None, :, :], median[None, :]  # Bolt shape: (batch, h, q)


class TestBackendGlue:
    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        pytest.importorskip("torch")

    @pytest.mark.parametrize("as_list", [False, True], ids=["bolt", "chronos2"])
    def test_context_crosses_as_float32_and_quantiles_come_back_as_h_by_q(
        self, as_list: bool
    ) -> None:
        import torch

        stub = _StubPipeline(as_list=as_list)
        backend = ChronosBackend(stub, checkpoint="x/y", revision="f" * 40, dtype="float32")
        ctx = realized_variance(100) * 1e4
        out = backend.forecast(ctx, 3)
        (call,) = stub.calls
        sent = call["inputs"][0]
        assert isinstance(sent, torch.Tensor) and sent.dtype == torch.float32
        assert np.allclose(sent.numpy(), ctx.astype(np.float32))
        assert call["prediction_length"] == 3
        assert call["levels"] == _LEVELS  # exactly the trained levels: no interpolation
        assert out.taus == tuple(_LEVELS)
        assert out.values.shape == (3, 9) and out.values.dtype == np.float64
        assert out.native_mean is None  # the pipeline's "mean" is the median; not recorded
        assert np.all(np.diff(out.values, axis=1) > 0)
        assert np.allclose(out.values[2] / out.values[0], 1.03 / 1.01)

    def test_identity_and_limits(self) -> None:
        stub = _StubPipeline(as_list=False)
        backend = ChronosBackend(
            stub, checkpoint="x/y", revision="f" * 40, dtype="bfloat16", versions={"torch": "t"}
        )
        assert backend.max_context == 2048
        assert backend.taus == tuple(_LEVELS)
        ident = backend.identity()
        assert ident["backend"] == "chronos"
        assert ident["pipeline"] == "_StubPipeline"
        assert ident["dtype"] == "bfloat16"
        assert ident["revision"] == "f" * 40
        assert ident["torch"] == "t"


@pytest.mark.tsfm
@pytest.mark.parametrize("checkpoint", ["amazon/chronos-bolt-small", "amazon/chronos-2"])
class TestRealCheckpoint:
    def test_spec_pins_the_weights(self, checkpoint: str) -> None:
        spec = Chronos(checkpoint=checkpoint).spec()
        assert _SHA.match(spec["revision"])
        assert spec["revision"] == resolve_hf_revision(checkpoint)
        assert spec["checkpoint"] == checkpoint
        assert spec["dtype"] == "float32"
        assert {"chronos_forecasting", "transformers", "torch"} <= set(spec)
        assert len(spec["quantile_levels"]) in {9, 21}

    def test_bit_identical_twice(self, checkpoint: str) -> None:
        rv = realized_variance(1000)
        a = Chronos(checkpoint=checkpoint).fit(rv)
        b = Chronos(checkpoint=checkpoint).fit(rv)
        assert np.array_equal(a.rv_forecast(5).values, b.rv_forecast(5).values)
        assert a.predict(1) == b.predict(1)
        assert a.predict(5) == b.predict(5)
        assert a.spec()["rv_forecasts"] == b.spec()["rv_forecasts"]

    def test_forecast_is_a_sane_daily_variance(self, checkpoint: str) -> None:
        rv = realized_variance(1000)
        fitted = Chronos(checkpoint=checkpoint).fit(rv)
        dist = fitted.predict(1)
        meta = fitted.spec()["rv_forecasts"]["1"]
        grid = np.asarray(meta["values"])
        assert meta["crossings_rearranged"] == 0 and meta["clipped_at_zero"] == 0
        assert grid[0] <= dist.variance() <= grid[-1]
        # sigma = sqrt(vhat); squaring it back is exact only to an ulp
        assert dist.variance() == pytest.approx(
            quantile_grid_mean(np.asarray(meta["taus"]), grid), rel=1e-12
        )
        level = float(np.mean(rv[-22:]))
        assert 0.3 * level < dist.variance() < 3.0 * level

    def test_scale_stable_at_the_default_units(self, checkpoint: str) -> None:
        rv = realized_variance(1000)
        v4 = Chronos(checkpoint=checkpoint, input_scale=1e4).fit(rv).predict(1).variance()
        v6 = Chronos(checkpoint=checkpoint, input_scale=1e6).fit(rv).predict(1).variance()
        assert v6 == pytest.approx(v4, rel=1e-3)

    def test_update_moves_with_the_context(self, checkpoint: str) -> None:
        rv = realized_variance(1100)
        fitted = Chronos(checkpoint=checkpoint, context_length=512).fit(rv[:1000])
        assert fitted.update(rv[:1000]).predict(1) == fitted.predict(1)
        shocked = rv[1:1001].copy()
        shocked[-1] *= 10.0
        assert fitted.update(shocked).predict(1).variance() > fitted.predict(1).variance()

    def test_first_step_does_not_depend_on_the_requested_horizon(self, checkpoint: str) -> None:
        fitted = Chronos(checkpoint=checkpoint).fit(realized_variance(1000))
        assert np.array_equal(fitted.rv_forecast(1).values[0], fitted.rv_forecast(5).values[0])
