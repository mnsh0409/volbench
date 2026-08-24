"""TimesFM 2.5 (Google) zero-shot adapter.

READ FIRST — the RV -> return-distribution mapping, the ``input_scale`` unit
convention, the quantile post-processing and the update-as-context-extension
semantics are all defined once in :mod:`volbench.models.tsfm_common`; this
module only loads a checkpoint and turns a context into a quantile grid. In
short: ``fit(rv)`` records the trailing context of a **realized-variance**
series; ``predict(h)`` takes the MEAN of TimesFM's predictive distribution of
RV at ``t+h`` as the variance forecast and emits ``Normal(0, sqrt(vhat))``
over the next-period return — the same shape HAR emits.

Checkpoint. ``google/timesfm-2.5-200m-pytorch`` through the ``timesfm``
package's ``TimesFM_2p5_200M_torch``. Its ``forecast`` returns, per step, a
10-vector: index 0 is the model's point head, indices 1..9 the quantiles at
0.1..0.9. The point head is recorded as ``native_mean`` (visible in the fitted
``spec()``) and, as for every adapter here, not scored. No sampling anywhere.

Forecast configuration (all in ``spec()``; the defaults follow the package's
own README recommendation):

- ``normalize_inputs``: reversible instance normalisation of the context.
- ``use_continuous_quantile_head``: the separate quantile head that avoids
  quantile collapse (horizons ≤ 1024).
- ``force_flip_invariance``: forecasts averaged with the sign-flipped
  context's forecast, making the model odd-symmetric.
- ``infer_is_positive``: a non-negative context yields non-negative
  quantiles — appropriate for a variance; the zero-clip in tsfm_common then
  never fires for this model.
- ``fix_quantile_crossing``: the package's own monotone repair, applied
  before tsfm_common's rearrangement (which then counts zero crossings).
- ``max_horizon``: the compiled decode horizon; ``predict(h)`` requires
  ``h <= max_horizon``.

The decode is "compiled" (a closure, not ``torch.compile`` — that is switched
off for determinism) with ``max_context`` rounded up to the 32-step patch and
re-compiled whenever the context length changes, so a context is never padded
against a stale, larger ``max_context``. TimesFM picks CUDA itself when it is
available; there is no device knob.

Identity. ``spec()`` carries the checkpoint id, the resolved commit hash of
its weights, the forecast configuration, and the ``timesfm`` / ``torch``
versions.
"""

from __future__ import annotations

import math
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

__all__ = ["DEFAULT_TIMESFM_CHECKPOINT", "TimesFM", "TimesFMBackend", "TimesFMForecastOptions"]

DEFAULT_TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


@dataclass(frozen=True)
class TimesFMForecastOptions:
    """The subset of ``timesfm.ForecastConfig`` this adapter exposes (all hashed)."""

    max_horizon: int = 128
    normalize_inputs: bool = True
    use_continuous_quantile_head: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_horizon": self.max_horizon,
            "normalize_inputs": self.normalize_inputs,
            "use_continuous_quantile_head": self.use_continuous_quantile_head,
            "force_flip_invariance": self.force_flip_invariance,
            "infer_is_positive": self.infer_is_positive,
            "fix_quantile_crossing": self.fix_quantile_crossing,
        }


class TimesFMBackend:
    """A loaded ``TimesFM_2p5_200M_torch`` behind :class:`~volbench.models.tsfm_common.TSFMBackend`.

    ``model`` needs ``compile(ForecastConfig)`` and ``forecast(horizon,
    inputs) -> (points, quantiles)`` with quantiles shaped ``(batch, h, 10)``,
    plus ``model.config`` carrying ``quantiles``, ``context_limit`` and
    ``input_patch_len`` — the CI tests hand in a stub with exactly that.
    """

    def __init__(
        self,
        model: Any,
        *,
        checkpoint: str,
        revision: str,
        options: TimesFMForecastOptions,
        versions: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._checkpoint = checkpoint
        self._revision = revision
        self._options = options
        self._versions = dict(versions or {})
        definition = model.model.config
        self._taus = tuple(float(q) for q in definition.quantiles)
        self._patch = int(definition.input_patch_len)
        self._max_context = int(definition.context_limit) - options.max_horizon
        self._compiled_context = -1

    @property
    def taus(self) -> tuple[float, ...]:
        return self._taus

    @property
    def max_context(self) -> int:
        return self._max_context

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "timesfm",
            "checkpoint": self._checkpoint,
            "revision": self._revision,
            "dtype": "float32",
            "torch_compile": False,
            "quantile_levels": list(self._taus),
            **self._options.as_dict(),
            **self._versions,
        }

    def _compile_for(self, n_context: int) -> None:
        import timesfm

        max_context = math.ceil(n_context / self._patch) * self._patch
        if max_context == self._compiled_context:
            return
        o = self._options
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=o.max_horizon,
                normalize_inputs=o.normalize_inputs,
                use_continuous_quantile_head=o.use_continuous_quantile_head,
                force_flip_invariance=o.force_flip_invariance,
                infer_is_positive=o.infer_is_positive,
                fix_quantile_crossing=o.fix_quantile_crossing,
            )
        )
        self._compiled_context = max_context

    def forecast(self, context: NDArray[np.float64], h: int) -> RVQuantileForecast:
        import torch

        if h > self._options.max_horizon:
            raise ValueError(f"h={h} exceeds max_horizon={self._options.max_horizon}")
        ctx = np.asarray(context, dtype=np.float64)
        self._compile_for(ctx.size)
        torch.manual_seed(0)  # no sampling path; see tsfm_common "Determinism"
        with torch.inference_mode():
            _points, quantiles = self._model.forecast(horizon=h, inputs=[ctx.copy()])
        arr = np.asarray(quantiles, dtype=np.float64)[0]  # (h, 10): point head, then q0.1..q0.9
        if arr.shape != (h, len(self._taus) + 1):
            raise RuntimeError(f"TimesFM returned shape {arr.shape}, expected {(h, 10)}")
        return RVQuantileForecast(taus=self._taus, values=arr[:, 1:], native_mean=arr[:, 0])


@cache
def _load_timesfm(
    checkpoint: str, revision: str | None, options: TimesFMForecastOptions
) -> TimesFMBackend:
    """One model per (checkpoint, revision, options) per process."""
    from importlib.metadata import version as pkg_version

    import timesfm
    import torch

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        checkpoint, revision=revision, torch_compile=False
    )
    model.model.eval()
    return TimesFMBackend(
        model,
        checkpoint=checkpoint,
        revision=resolve_hf_revision(checkpoint, revision),
        options=options,
        versions={"timesfm": pkg_version("timesfm"), "torch": torch.__version__},
    )


@dataclass(frozen=True)
class TimesFM(ZeroShotRVModel):
    """TimesFM 2.5, zero-shot on a realized-variance context.

    Parameters
    ----------
    checkpoint, revision:
        Hugging Face model id and an optional pin; the resolved commit hash
        always enters ``spec()``.
    options:
        :class:`TimesFMForecastOptions`; every field is hashed.
    context_length, input_scale, backend:
        See :class:`~volbench.models.tsfm_common.ZeroShotRVModel`.
    """

    checkpoint: str = DEFAULT_TIMESFM_CHECKPOINT
    revision: str | None = None
    options: TimesFMForecastOptions = TimesFMForecastOptions()

    @property
    def name(self) -> str:
        return checkpoint_slug("timesfm", self.checkpoint)

    def _load_backend(self) -> TSFMBackend:
        return _load_timesfm(self.checkpoint, self.revision, self.options)
