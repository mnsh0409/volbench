"""Daily variance proxies — all DAILY units, never annualized (CLAUDE.md rule 2).

Every function here is pure: it takes explicit arrays/Series and returns a new
Series, with no hidden state and no reference to any other day's data than the
inputs given. Each proxy is contemporaneous — it estimates day t's variance
from day t's own price data (the realized target itself, not a forecast of
it), so nothing here reads across the train/test boundary; that guarantee is
independent of, and does not replace, RollingOriginSplitter.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

__all__ = [
    "garman_klass",
    "log_returns",
    "overnight_plus_range_variance",
    "parkinson",
    "realized_variance_from_bars",
    "squared_return",
]

_LN2 = math.log(2.0)
_GK_CLOSE_COEF = 2.0 * _LN2 - 1.0


def _log(values: pd.Series) -> pd.Series:
    """``np.log`` of a Series *is* a Series (pandas implements ``__array_ufunc__``);
    numpy's stubs only know it as an ndarray. Same call, correct type."""
    return cast(pd.Series, np.log(values))


def log_returns(close: pd.Series) -> pd.Series:
    """Close-to-close log returns: ``r_t = ln(C_t / C_{t-1})``, daily units.

    Added at M1 integration: the models and the evaluator both speak in
    *returns* (a model's ``predict`` is a distribution over the next return,
    and CRPS/pinball/VaR are scored against the realized return), but the data
    layer previously exposed only ``r_t^2``. Squaring and un-squaring loses the
    sign, so every caller was rolling its own ``np.diff(np.log(...))`` — which
    is exactly the hand-rolled index arithmetic this project tries not to have.

    The first observation is NaN — there is no ``C_{t-1}`` for it — and is left
    as a gap rather than dropped or filled, so the output stays index-aligned
    with the input. That alignment is load-bearing: ``run_backtest`` matches
    returns to proxies positionally, so a helper that silently shortened the
    series by one would offset every forecast against the wrong realization.
    """
    c = close.astype(np.float64)
    out: pd.Series = (_log(c) - _log(c.shift(1))).rename("log_return")
    return out


def squared_return(close: pd.Series) -> pd.Series:
    """Squared close-to-close log return: ``r_t^2`` with ``r_t = ln(C_t / C_{t-1})``.

    An unbiased but noisy daily-variance proxy (Patton, 2011). The first
    observation is NaN — there is no ``C_{t-1}`` for it, and it is left as a
    gap rather than dropped or filled so the output stays index-aligned with
    the input.
    """
    out: pd.Series = (log_returns(close) ** 2).rename("squared_return")
    return out


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) range-based daily variance proxy: ``(ln(H/L))^2 / (4 ln 2)``.

    Uses only day t's own high/low, so it is exact per day up to the
    estimator's known bias (ignores drift and overnight gaps).
    """
    h = high.astype(np.float64)
    lo = low.astype(np.float64)
    if (h < lo).any():
        raise ValueError("high must be >= low at every observation")
    hl = _log(h / lo)
    out: pd.Series = (hl**2 / (4.0 * _LN2)).rename("parkinson")
    return out


def garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Garman-Klass (1980) daily variance proxy.

    ``0.5*(ln(H/L))^2 - (2 ln 2 - 1)*(ln(C/O))^2``. Non-negative for any
    genuine OHLC bar: since L <= O, C <= H by construction, ``ln(H/L) >=
    |ln(C/O)|``, and with the 0.5 vs. ~0.386 coefficients that inequality
    forces the difference to be >= 0.
    """
    o = open_.astype(np.float64)
    h = high.astype(np.float64)
    lo = low.astype(np.float64)
    c = close.astype(np.float64)
    if (h < lo).any():
        raise ValueError("high must be >= low at every observation")
    hl_term = _log(h / lo) ** 2
    co_term = _log(c / o) ** 2
    out: pd.Series = (0.5 * hl_term - _GK_CLOSE_COEF * co_term).rename("garman_klass")
    return out


def _check_ohlc(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> None:
    """A bar whose open or close lies outside [low, high] is not a bar.

    Stricter than the range proxies above, which only check ``high >= low``:
    the Rogers-Satchell terms are products of two logs that share a sign
    only when the open and close sit inside the range, so an inconsistent
    bar can turn the estimator negative — and HAR takes its log.
    """
    if (high < low).any():
        raise ValueError("high must be >= low at every observation")
    if (high < open_).any() or (high < close).any():
        raise ValueError("high must be >= open and >= close at every observation")
    if (low > open_).any() or (low > close).any():
        raise ValueError("low must be <= open and <= close at every observation")


def _rogers_satchell(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Rogers & Satchell (1991) per-day estimator of the open-to-close variance.

    ``ln(H/O) ln(H/C) + ln(L/O) ln(L/C)``. Unbiased for the diffusion variance
    of a Brownian motion with *any* drift (their result), which Parkinson and
    Garman-Klass are not; blind, by construction, to what happens between the
    previous close and today's open.
    """
    hi_open, hi_close = _log(high / open_), _log(high / close)
    lo_open, lo_close = _log(low / open_), _log(low / close)
    out: pd.Series = hi_open * hi_close + lo_open * lo_close
    return out


def overnight_plus_range_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Per-day close-to-close variance estimate: squared overnight jump plus Rogers-Satchell.

    ``(ln(O_t / C_{t-1}))^2 + ln(H_t/O_t) ln(H_t/C_t) + ln(L_t/O_t) ln(L_t/C_t)``,
    daily units. The first observation is NaN — there is no ``C_{t-1}`` for
    it — and is left as a gap so the output stays index-aligned with its
    inputs (the same convention as :func:`squared_return`).

    Why it exists (M1 report §4.4): a range proxy — Parkinson, Garman-Klass,
    Rogers-Satchell — estimates the variance *between the open and the close*.
    The quantity every volbench model forecasts, and is scored on, is the
    variance of the close-to-close return, which also contains the overnight
    jump ``ln(O_t / C_{t-1})``. Feeding HAR a range proxy and scoring it
    against close-to-close returns therefore biased its variance forecast low
    by the overnight share, independent of any model error. This target puts
    the two pieces back together, day by day: the squared overnight jump is an
    unbiased (if noisy) estimate of the overnight variance, and Rogers-Satchell
    is the intraday estimator that stays unbiased under drift.

    What it is *not*: Yang & Zhang (2000). YZ is a **windowed** estimator over
    ``n`` days — ``sigma_o^2 + k sigma_c^2 + (1 - k) sigma_RS^2`` with
    ``sigma_o^2``/``sigma_c^2`` the *demeaned sample variances* of the
    overnight and open-to-close returns across the window, ``sigma_RS^2`` the
    window average of RS, and ``k = 0.34 / (1.34 + (n + 1) / (n - 1))``.
    That is an excellent *volatility* estimator and the wrong *target*: a
    forecast for day ``t`` must be scored against day ``t``'s own realization,
    and a window ending after ``t`` would put the future into the target.
    Demeaning across a window is also what makes YZ drift-independent in the
    overnight term; per day there is nothing to demean, so this estimator is
    unbiased under zero drift — the standard daily-return assumption every
    baseline here already makes.

    Sources. Rogers, L. C. G. & Satchell, S. E. (1991), "Estimating Variance
    From High, Low and Closing Prices", *Annals of Applied Probability* 1(4),
    504-512, doi:10.1214/aoap/1177005835. Yang, D. & Zhang, Q. (2000),
    "Drift-Independent Volatility Estimation Based on High, Low, Open, and
    Close Prices", *Journal of Business* 73(3), 477-492. Both full texts are
    paywalled; the formulas were corroborated (2026-08-24) across CRAN's TTR
    package documentation (``volatility``, calc="rogers.satchell" and
    "yang.zhang"), arXiv:1803.07152 §2, and portfoliooptimizer.io's
    range-estimator overview, which agree with each other and with the
    Project Euclid abstract's drift-independence claim.

    Bars must be consistent: ``low <= min(open, close) <= max(open, close) <=
    high``. Anything else raises rather than yielding a negative variance.
    """
    o = open_.astype(np.float64)
    h = high.astype(np.float64)
    lo = low.astype(np.float64)
    c = close.astype(np.float64)
    _check_ohlc(o, h, lo, c)
    overnight = _log(o / c.shift(1))  # NaN on the first day: no previous close
    out: pd.Series = (overnight**2 + _rogers_satchell(o, h, lo, c)).rename(
        "overnight_plus_range"
    )
    return out


def realized_variance_from_bars(bars: pd.Series, *, min_bars: int = 2) -> pd.Series:
    """Daily realized variance from intraday price bars.

    ``RV_t = sum`` of squared intraday log returns computed strictly within
    calendar day t (UTC) — a return spanning midnight (the last bar of one
    day to the first bar of the next) is never counted, so no information
    from day t+1 can enter day t's estimate. A day with fewer than
    ``min_bars`` price observations (hence fewer than ``min_bars - 1``
    intraday returns) is reported as NaN rather than a noisy near-zero
    estimate.

    Parameters
    ----------
    bars:
        Strictly positive prices indexed by a tz-aware, strictly increasing
        :class:`~pandas.DatetimeIndex` (any intraday frequency).
    min_bars:
        Minimum number of price observations required in a day (>= 2, since
        one return needs two prices) for that day's RV to be reported.
    """
    if min_bars < 2:
        raise ValueError("min_bars must be >= 2 (need at least one intraday return)")
    index = bars.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("bars must be indexed by a DatetimeIndex")
    if index.tz is None:
        raise ValueError("bars index must be tz-aware (UTC)")
    if not index.is_monotonic_increasing:
        raise ValueError("bars index must be strictly increasing")
    if bars.isna().any():
        raise ValueError("bars must not contain NaN")
    if (bars.to_numpy(dtype=np.float64) <= 0.0).any():
        raise ValueError("bars must be strictly positive prices")

    if bars.size == 0:
        return pd.Series(dtype=np.float64, name="realized_variance")

    day = index.tz_convert("UTC").floor("D").to_numpy()
    log_px = np.log(bars.to_numpy(dtype=np.float64))

    is_new_day = np.empty(log_px.size, dtype=bool)
    is_new_day[0] = True
    is_new_day[1:] = day[1:] != day[:-1]

    intraday_return = np.diff(log_px, prepend=np.nan)
    intraday_return[is_new_day] = np.nan

    sq_return = pd.Series(intraday_return**2)
    rv = sq_return.groupby(day).sum()
    counts = pd.Series(np.ones(log_px.size, dtype=np.int64)).groupby(day).sum()

    rv = rv.where(counts >= min_bars, other=np.nan)
    rv.index.name = "date"
    out: pd.Series = rv.rename("realized_variance")
    return out
