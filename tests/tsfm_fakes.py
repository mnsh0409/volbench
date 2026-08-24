"""Deterministic stand-ins for the TSFM backends — the CI half of the tests.

No weights, no torch, no network. ``FakeBackend`` behaves like a foundation
model in the ways the adapter contract cares about (it depends on the context
it is handed, it is scale-equivariant, its rows differ by horizon) and it
records every context it sees, which is what the context-construction and
leakage assertions read. ``ScriptedBackend`` returns exactly the rows it is
given, for the crossing / clipping / shape checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from volbench.models.tsfm_common import RVQuantileForecast

DEFAULT_TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FAKE_SHA = "f" * 40


class FakeBackend:
    """Lognormal quantiles around the trailing-22 mean of the context.

    ``values[k] = level * exp(sigma_log * z_tau) * (1 + 0.01 * (k + 1))`` —
    scale-equivariant in the context (so ``input_scale`` round-trips exactly
    up to floating point), and different at every horizon.
    """

    def __init__(
        self,
        taus: tuple[float, ...] = DEFAULT_TAUS,
        *,
        max_context: int = 2048,
        native_mean: bool = False,
        sigma_log: float = 0.5,
        identity: dict[str, Any] | None = None,
    ) -> None:
        self._taus = tuple(float(t) for t in taus)
        self._max_context = max_context
        self._native = native_mean
        self._sigma = sigma_log
        self._identity = dict(
            identity or {"backend": "fake", "checkpoint": "fake/echo", "revision": FAKE_SHA}
        )
        self.calls: list[tuple[NDArray[np.float64], int]] = []

    @property
    def taus(self) -> tuple[float, ...]:
        return self._taus

    @property
    def max_context(self) -> int:
        return self._max_context

    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    def level_of(self, context: NDArray[np.float64]) -> float:
        return float(np.mean(np.asarray(context, dtype=np.float64)[-22:]))

    def forecast(self, context: NDArray[np.float64], h: int) -> RVQuantileForecast:
        ctx = np.asarray(context, dtype=np.float64)
        self.calls.append((ctx.copy(), h))
        level = self.level_of(ctx)
        z = stats.norm.ppf(np.asarray(self._taus))
        steps = 1.0 + 0.01 * np.arange(1, h + 1, dtype=np.float64)
        values = level * np.exp(self._sigma * z)[None, :] * steps[:, None]
        native = level * np.exp(0.5 * self._sigma**2) * steps if self._native else None
        return RVQuantileForecast(taus=self._taus, values=values, native_mean=native)


class ScriptedBackend:
    """Returns ``rows[:h]`` verbatim (``rows`` shaped ``(H, len(taus))``)."""

    def __init__(
        self,
        rows: NDArray[np.float64],
        taus: tuple[float, ...] = DEFAULT_TAUS,
        *,
        max_context: int = 2048,
    ) -> None:
        self.rows = np.asarray(rows, dtype=np.float64)
        self._taus = tuple(float(t) for t in taus)
        self._max_context = max_context

    @property
    def taus(self) -> tuple[float, ...]:
        return self._taus

    @property
    def max_context(self) -> int:
        return self._max_context

    def identity(self) -> dict[str, Any]:
        return {"backend": "scripted", "checkpoint": "fake/scripted", "revision": FAKE_SHA}

    def forecast(self, context: NDArray[np.float64], h: int) -> RVQuantileForecast:
        return RVQuantileForecast(taus=self._taus, values=self.rows[:h].copy())


def realized_variance(n: int = 600, seed: int = 0) -> NDArray[np.float64]:
    """A lognormal daily-variance series at the ~1e-4 level, like the other model tests."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(np.log(1e-4), 0.4, size=n))
