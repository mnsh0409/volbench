"""Chronos-Bolt / Chronos-2 (Amazon) zero-shot adapter.

READ FIRST — the RV -> return-distribution mapping, the ``input_scale`` unit
convention, the quantile post-processing and the update-as-context-extension
semantics are all defined once in :mod:`volbench.models.tsfm_common`; this
module only loads a checkpoint and turns a context into a quantile grid. In
short: ``fit(rv)`` records the trailing context of a **realized-variance**
series; ``predict(h)`` takes the MEAN of Chronos's predictive distribution of
RV at ``t+h`` as the variance forecast and emits ``Normal(0, sqrt(vhat))``
over the next-period return — the same shape HAR emits.

Checkpoints. ``chronos-forecasting`` dispatches on the config, so one adapter
covers both families:

- ``amazon/chronos-bolt-{tiny,mini,small,base}`` — direct multi-step quantile
  output at the nine levels 0.1..0.9, 2048-step context, 64-step direct
  horizon (longer horizons roll the quantile grid forward as pseudo-samples).
  **Default**, per the brief's preference for direct-quantile variants.
- ``amazon/chronos-2`` — the 2025 successor, 21 levels 0.01..0.99, 8192-step
  context; also direct quantiles.

Neither samples. ``predict_quantiles`` is asked for exactly the levels the
checkpoint was trained on, so no interpolation happens inside the pipeline.
Note that what the pipeline returns as ``mean`` is the 0.5 quantile (its own
source says so); the adapter ignores it and, like every TSFM here, scores the
mean of the grid.

Identity. ``spec()`` carries the checkpoint id, the resolved commit hash of
its weights, the pipeline class, the dtype, and the ``chronos-forecasting``
/ ``transformers`` / ``torch`` versions — any of which can move a number.
``device`` is deliberately *not* in ``spec()``: it is where the same
computation runs, not what is computed; float32 results on CPU and GPU agree
to rounding, not bit for bit, which the tsfm-marked tests document.

Cost. One forward pass per origin per horizon (the fitted object memoises per
``h``), regardless of ``refit_every``; see tsfm_common.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volbench.models.tsfm_common import (
    RVQuantileForecast,
    TSFMBackend,
    ZeroShotRVModel,
    checkpoint_slug,
    resolve_hf_revision,
)

__all__ = ["DEFAULT_CHRONOS_CHECKPOINT", "Chronos", "ChronosBackend"]

DEFAULT_CHRONOS_CHECKPOINT = "amazon/chronos-bolt-small"


def _torch_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class ChronosBackend:
    """A loaded ``chronos`` pipeline behind :class:`~volbench.models.tsfm_common.TSFMBackend`.

    ``pipeline`` is any object with the ``BaseChronosPipeline`` surface used
    here (``quantiles``, ``model_context_length``, ``predict_quantiles``) —
    the CI tests hand in a stub; the tsfm-marked tests the real thing.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        checkpoint: str,
        revision: str,
        dtype: str,
        versions: dict[str, str] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._checkpoint = checkpoint
        self._revision = revision
        self._dtype = dtype
        self._versions = dict(versions or {})
        self._taus = tuple(float(q) for q in pipeline.quantiles)

    @property
    def taus(self) -> tuple[float, ...]:
        return self._taus

    @property
    def max_context(self) -> int:
        return int(self._pipeline.model_context_length)

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "chronos",
            "checkpoint": self._checkpoint,
            "revision": self._revision,
            "pipeline": type(self._pipeline).__name__,
            "dtype": self._dtype,
            "quantile_levels": list(self._taus),
            **self._versions,
        }

    def forecast(self, context: NDArray[np.float64], h: int) -> RVQuantileForecast:
        import torch

        # No sampling path exists in these pipelines; seeding makes that an
        # assumption the bit-identity tests can falsify rather than trust.
        torch.manual_seed(0)
        ctx = torch.tensor(np.asarray(context, dtype=np.float64), dtype=torch.float32)
        with torch.inference_mode():
            quantiles, _median = self._pipeline.predict_quantiles(
                [ctx], prediction_length=h, quantile_levels=list(self._taus)
            )
        first = quantiles[0]  # Bolt: tensor (batch, h, q) -> (h, q); Chronos-2: list -> (1, h, q)
        arr = first.detach().to(torch.float32).cpu().numpy().astype(np.float64)
        return RVQuantileForecast(taus=self._taus, values=arr.reshape(-1, len(self._taus)))


@cache
def _load_chronos(checkpoint: str, revision: str | None, dtype: str, device: str) -> ChronosBackend:
    """One pipeline per (checkpoint, revision, dtype, device) per process."""
    import chronos
    import torch
    import transformers
    from chronos import BaseChronosPipeline

    kwargs: dict[str, Any] = {"device_map": _torch_device(device), "dtype": dtype}
    if revision is not None:
        kwargs["revision"] = revision
    pipeline = BaseChronosPipeline.from_pretrained(checkpoint, **kwargs)
    pipeline.model.eval()
    return ChronosBackend(
        pipeline,
        checkpoint=checkpoint,
        revision=resolve_hf_revision(checkpoint, revision),
        dtype=dtype,
        versions={
            "chronos_forecasting": chronos.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
    )


@dataclass(frozen=True)
class Chronos(ZeroShotRVModel):
    """Chronos-Bolt / Chronos-2, zero-shot on a realized-variance context.

    Parameters
    ----------
    checkpoint:
        Hugging Face model id (Bolt by default; ``amazon/chronos-2`` works too).
    revision:
        Branch, tag or commit to pin; ``None`` = the cached ``main``. The
        resolved commit hash always enters ``spec()``.
    dtype:
        ``"float32"`` (default, and what the determinism tests pin) or
        ``"bfloat16"`` for speed at the cost of precision.
    device:
        ``"auto"`` (CUDA if available), ``"cuda"``, ``"cpu"``. Not in ``spec()``.
    context_length, input_scale, backend:
        See :class:`~volbench.models.tsfm_common.ZeroShotRVModel`.
    """

    checkpoint: str = DEFAULT_CHRONOS_CHECKPOINT
    revision: str | None = None
    dtype: str = "float32"
    device: str = "auto"

    @property
    def name(self) -> str:
        return checkpoint_slug("chronos", self.checkpoint)

    def _load_backend(self) -> TSFMBackend:
        return _load_chronos(self.checkpoint, self.revision, self.dtype, self.device)
