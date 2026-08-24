"""TimeGPT (Nixtla, hosted API) zero-shot adapter — opt-in, key-gated, never in CI.

READ FIRST — the RV -> return-distribution mapping, the ``input_scale`` unit
convention, the quantile post-processing and the update-as-context-extension
semantics are all defined once in :mod:`volbench.models.tsfm_common`; this
module only wraps the API call. In short: ``fit(rv)`` records the trailing
context of a **realized-variance** series; ``predict(h)`` takes the MEAN of
TimeGPT's predictive distribution of RV at ``t+h`` as the variance forecast
and emits ``Normal(0, sqrt(vhat))`` over the next-period return — the same
shape HAR emits.

Gating — three independent conditions, all required before a byte leaves
the machine:

1. ``TimeGPT(enabled=True)`` — an explicit constructor opt-in. The default
   ``enabled=False`` refuses in ``fit`` **before** a client exists, so a
   TimeGPT cell in a grid config can never phone home by accident.
2. ``NIXTLA_API_KEY`` in the environment — the only place a key may live
   (CLAUDE.md: never in code, config or fixtures). The adapter reads it
   itself; it is not a constructor argument, so it cannot end up in a spec,
   a hash or a results file.
3. Its tests carry ``@pytest.mark.timegpt``: skipped unless the key is set,
   and always skipped under ``CI`` (tests/conftest.py).

The research design lists TimeGPT behind an API-key flag and *excluded from
the headline if access is unstable*; this adapter is that flag.

Request shape. The context is sent as one series with an integer ``ds``
(``0..n-1``, ``freq=1``): the series has no calendar the model may use, and
positions map to the volbench index monotonically, so nothing after the
origin can enter the request. ``quantiles`` asks for the nine levels
0.1..0.9 by default; the response's ``TimeGPT`` point column is recorded as
``native_mean`` (Nixtla calls it the mean) and, as for every adapter here,
not scored.

Identity and determinism — the honest caveats. A hosted model has no commit
hash: ``spec()`` records the API model id and the ``nixtla`` client version,
and *cannot* pin the remote weights. Whether the service returns the same
forecast for the same request twice is checked by the key-gated test, not
guaranteed by construction. Both are reasons the design keeps TimeGPT out of
the headline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volbench.models.tsfm_common import (
    RVQuantileForecast,
    TSFMBackend,
    ZeroShotRVModel,
    checkpoint_slug,
)

__all__ = ["DEFAULT_TIMEGPT_MODEL", "TIMEGPT_KEY_ENV", "TimeGPT", "TimeGPTBackend"]

DEFAULT_TIMEGPT_MODEL = "timegpt-1"
TIMEGPT_KEY_ENV = "NIXTLA_API_KEY"
DEFAULT_TIMEGPT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: TimeGPT accepts long histories; the adapter's cap is the protocol window
#: size, not a model limit (``context_length`` on the model lowers it).
_MAX_CONTEXT = 100_000


class TimeGPTBackend:
    """A ``NixtlaClient`` behind :class:`~volbench.models.tsfm_common.TSFMBackend`.

    ``client`` needs ``forecast(df=, h=, freq=, quantiles=, model=)``
    returning a frame with ``ds``, ``TimeGPT`` and ``TimeGPT-q-<level>``
    columns — the CI tests hand in a stub; only the key-gated test uses the
    real client.
    """

    def __init__(
        self,
        client: Any,
        *,
        api_model: str,
        quantiles: tuple[float, ...],
        versions: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._api_model = api_model
        self._taus = tuple(sorted(float(q) for q in quantiles))
        self._versions = dict(versions or {})

    @property
    def taus(self) -> tuple[float, ...]:
        return self._taus

    @property
    def max_context(self) -> int:
        return _MAX_CONTEXT

    def identity(self) -> dict[str, Any]:
        return _timegpt_identity(self._api_model, self._taus, self._versions)

    def forecast(self, context: NDArray[np.float64], h: int) -> RVQuantileForecast:
        import pandas as pd

        ctx = np.asarray(context, dtype=np.float64)
        frame = pd.DataFrame(
            {"unique_id": "rv", "ds": np.arange(ctx.size, dtype=np.int64), "y": ctx}
        )
        out = self._client.forecast(
            df=frame, h=h, freq=1, quantiles=list(self._taus), model=self._api_model
        )
        out = out.sort_values("ds").reset_index(drop=True)
        if len(out) != h:
            raise RuntimeError(f"TimeGPT returned {len(out)} rows for h={h}")
        expected_ds = np.arange(ctx.size, ctx.size + h, dtype=np.int64)
        if not np.array_equal(out["ds"].to_numpy(dtype=np.int64), expected_ds):
            raise RuntimeError("TimeGPT response is not the h steps after the context end")
        cols = [f"TimeGPT-q-{round(q * 100)}" for q in self._taus]
        values = out[cols].to_numpy(dtype=np.float64)
        native = out["TimeGPT"].to_numpy(dtype=np.float64) if "TimeGPT" in out else None
        return RVQuantileForecast(taus=self._taus, values=values, native_mean=native)


def _timegpt_identity(
    api_model: str, taus: tuple[float, ...], versions: dict[str, str]
) -> dict[str, Any]:
    return {
        "backend": "timegpt",
        "checkpoint": api_model,
        "revision": "hosted-api-unpinned",
        "quantile_levels": list(taus),
        **versions,
    }


def _nixtla_version() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return {"nixtla": version("nixtla")}
    except PackageNotFoundError:
        return {"nixtla": "not-installed"}


@dataclass(frozen=True)
class TimeGPT(ZeroShotRVModel):
    """TimeGPT via Nixtla's API, zero-shot on a realized-variance context.

    Parameters
    ----------
    api_model:
        Nixtla model id (``timegpt-1`` by default; the client also accepts
        e.g. ``timegpt-1-long-horizon`` and the ``timegpt-2*`` family).
    quantiles:
        Levels requested from the API; the grid the mean is taken over.
    enabled:
        Explicit opt-in. ``False`` (default) makes ``fit`` raise before any
        client is built. ``spec()`` never needs the key or the network.
    context_length, input_scale, backend:
        See :class:`~volbench.models.tsfm_common.ZeroShotRVModel`.
    """

    api_model: str = DEFAULT_TIMEGPT_MODEL
    quantiles: tuple[float, ...] = DEFAULT_TIMEGPT_QUANTILES
    enabled: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.quantiles) < 2 or not all(0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError("quantiles must hold at least two levels strictly inside (0, 1)")

    @property
    def name(self) -> str:
        return checkpoint_slug("timegpt", self.api_model)

    def _identity(self) -> dict[str, Any]:
        if self.backend is not None:
            return self.backend.identity()
        return _timegpt_identity(self.api_model, tuple(sorted(self.quantiles)), _nixtla_version())

    def _load_backend(self) -> TSFMBackend:
        if not self.enabled:
            raise RuntimeError(
                "TimeGPT is opt-in: construct TimeGPT(enabled=True) to allow API calls"
            )
        key = os.environ.get(TIMEGPT_KEY_ENV)
        if not key:
            raise RuntimeError(f"TimeGPT needs the {TIMEGPT_KEY_ENV} environment variable")
        from nixtla import NixtlaClient

        client = NixtlaClient(api_key=key)
        return TimeGPTBackend(
            client, api_model=self.api_model, quantiles=self.quantiles, versions=_nixtla_version()
        )
