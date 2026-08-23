"""Point-forecast loss functions robust to noisy volatility proxies.

Only proxy-robust losses (Patton, 2011) belong here: with an unbiased but
noisy volatility proxy, MSE and QLIKE rank models consistently; most other
point losses do not. Probabilistic scores live on the Distribution object.

All losses are negatively oriented (smaller is better).
"""

from __future__ import annotations

import math

__all__ = ["mse", "pinball", "qlike"]


def qlike(forecast_var: float, proxy_var: float) -> float:
    """QLIKE loss: ``p/f - log(p/f) - 1`` for forecast variance f, proxy p.

    Minimized (at 0) when ``f == p``; heavily penalizes under-prediction of
    variance, which is the risk-relevant direction.
    """
    if forecast_var <= 0.0 or proxy_var <= 0.0:
        raise ValueError("variances must be strictly positive")
    r = proxy_var / forecast_var
    return r - math.log(r) - 1.0


def mse(forecast_var: float, proxy_var: float) -> float:
    """Squared error on the variance scale (proxy-robust; heavy-tailed in practice)."""
    d = forecast_var - proxy_var
    return d * d


def pinball(y: float, q: float, tau: float) -> float:
    """Quantile (pinball) loss of quantile forecast ``q`` at level ``tau``."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly inside (0, 1)")
    u = y - q
    return tau * u if u >= 0.0 else (tau - 1.0) * u
