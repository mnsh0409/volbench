"""Moirai 2.0 (Salesforce) zero-shot adapter.

READ FIRST — the RV -> return-distribution mapping, the ``input_scale`` unit
convention, the quantile post-processing and the update-as-context-extension
semantics are all defined once in :mod:`volbench.models.tsfm_common`; this
module only loads a checkpoint and turns a context into a quantile grid. In
short: ``fit(rv)`` records the trailing context of a **realized-variance**
series; ``predict(h)`` takes the MEAN of Moirai's predictive distribution of
RV at ``t+h`` as the variance forecast and emits ``Normal(0, sqrt(vhat))``
over the next-period return — the same shape HAR emits.

Checkpoint. ``Salesforce/moirai-2.0-R-small`` through ``uni2ts``'s
``Moirai2Module`` (weights) and ``Moirai2Forecast`` (the inference wrapper).
Moirai 2.0 is the decoder-only, quantile-loss successor of Moirai 1.x: it
emits the nine quantiles 0.1..0.9 directly, 16-step patches, four direct
prediction patches (64 steps) and autoregressive roll-out beyond. No sampling.
The adapter calls the wrapper's array-level ``predict`` — never the
``gluonts`` predictor path — so nothing here depends on a pandas calendar.

Units, and why ``input_scale`` matters most here: Moirai's ``PackedStdScaler``
computes ``sqrt(var + 1e-5)``. For a daily-variance series (level ~1e-4,
variance of the series ~1e-9) that epsilon *is* the scale, the standardised
context is flat, and the quantiles collapse onto the level (and cross). At
the default ``input_scale=1e4`` the epsilon is negligible and forecasts are
scale-stable from there upward; the tsfm-marked test pins that.

Identity. ``spec()`` carries the checkpoint id, the resolved commit hash of
its weights, the patch size, and the ``uni2ts`` / ``torch`` versions.
``device`` is not in ``spec()`` (see tsfm_chronos for why).
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

__all__ = ["DEFAULT_MOIRAI_CHECKPOINT", "Moirai", "MoiraiBackend"]

DEFAULT_MOIRAI_CHECKPOINT = "Salesforce/moirai-2.0-R-small"

#: Prediction tokens reserved out of the model's ``max_seq_len`` so that a
#: maximal context plus a horizon of up to this many patches still fits.
_RESERVED_PREDICT_TOKENS = 8


def _torch_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class MoiraiBackend:
    """A loaded ``Moirai2Forecast`` behind :class:`~volbench.models.tsfm_common.TSFMBackend`.

    ``forecaster`` needs ``hparams_context(prediction_length=, context_length=)``
    and ``predict([context]) -> (batch, n_quantiles, h)``, plus ``module`` with
    ``quantile_levels``, ``patch_size`` and ``max_seq_len`` — the CI tests
    hand in a stub with exactly that.
    """

    def __init__(
        self,
        forecaster: Any,
        *,
        checkpoint: str,
        revision: str,
        versions: dict[str, str] | None = None,
    ) -> None:
        self._forecaster = forecaster
        self._checkpoint = checkpoint
        self._revision = revision
        self._versions = dict(versions or {})
        module = forecaster.module
        self._taus = tuple(float(q) for q in module.quantile_levels)
        self._patch = int(module.patch_size)
        self._max_context = (int(module.max_seq_len) - _RESERVED_PREDICT_TOKENS) * self._patch

    @property
    def taus(self) -> tuple[float, ...]:
        return self._taus

    @property
    def max_context(self) -> int:
        return self._max_context

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "moirai",
            "checkpoint": self._checkpoint,
            "revision": self._revision,
            "dtype": "float32",
            "patch_size": self._patch,
            "quantile_levels": list(self._taus),
            **self._versions,
        }

    def forecast(self, context: NDArray[np.float64], h: int) -> RVQuantileForecast:
        import torch

        if h > _RESERVED_PREDICT_TOKENS * self._patch:
            raise ValueError(f"h={h} exceeds {_RESERVED_PREDICT_TOKENS * self._patch}")
        ctx = np.asarray(context, dtype=np.float64)
        torch.manual_seed(0)  # no sampling path; see tsfm_common "Determinism"
        with (
            torch.inference_mode(),
            self._forecaster.hparams_context(prediction_length=h, context_length=int(ctx.size)),
        ):
            out = self._forecaster.predict([ctx.astype(np.float32)])
        arr = np.asarray(out, dtype=np.float64)[0]  # (n_quantiles, h)
        if arr.shape != (len(self._taus), h):
            raise RuntimeError(
                f"Moirai returned shape {arr.shape}, expected {(len(self._taus), h)}"
            )
        return RVQuantileForecast(taus=self._taus, values=arr.T.copy())


@cache
def _load_moirai(checkpoint: str, revision: str | None, device: str) -> MoiraiBackend:
    """One forecaster per (checkpoint, revision, device) per process."""
    import torch
    import uni2ts
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    module = Moirai2Module.from_pretrained(checkpoint, revision=revision)
    forecaster = Moirai2Forecast(
        prediction_length=1,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=math.ceil(1000 / module.patch_size) * module.patch_size,
        module=module,
    ).to(_torch_device(device))
    forecaster.eval()
    return MoiraiBackend(
        forecaster,
        checkpoint=checkpoint,
        revision=resolve_hf_revision(checkpoint, revision),
        versions={"uni2ts": uni2ts.__version__, "torch": torch.__version__},
    )


@dataclass(frozen=True)
class Moirai(ZeroShotRVModel):
    """Moirai 2.0, zero-shot on a realized-variance context.

    Parameters
    ----------
    checkpoint, revision:
        Hugging Face model id and an optional pin; the resolved commit hash
        always enters ``spec()``.
    device:
        ``"auto"`` (CUDA if available), ``"cuda"``, ``"cpu"``. Not in ``spec()``.
    context_length, input_scale, backend:
        See :class:`~volbench.models.tsfm_common.ZeroShotRVModel`.
    """

    checkpoint: str = DEFAULT_MOIRAI_CHECKPOINT
    revision: str | None = None
    device: str = "auto"

    @property
    def name(self) -> str:
        return checkpoint_slug("moirai", self.checkpoint)

    def _load_backend(self) -> TSFMBackend:
        return _load_moirai(self.checkpoint, self.revision, self.device)
