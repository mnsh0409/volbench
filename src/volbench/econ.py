"""Economic value: the volatility-targeting backtest (docs/research_design.md).

The last metric in the research design, and the only one that asks what a
forecast is *worth* rather than how accurate it is. A volatility forecast that
wins on QLIKE but cannot size a position is a statistical result, not a
finance one; this module is the finance one.

    position_t = min(target_vol / forecast_vol_t, leverage_cap)

held into ``t+1`` and earning ``t+1``'s return, net of transaction costs on the
turnover ``|position_t - position_{t-1}|``.

What this module consumes, and what it must never do
====================================================
It reads **stored forecast rows** — a ``ResultsStore`` fragment, or
:func:`volbench.runner.read_grid_results` over a manifest — and nothing else.
It never runs a model, never touches a splitter, and never imports
:mod:`volbench.evaluate`. That is a deliberate structural boundary, not a
style choice: economic value is computed *after* the fact from a table whose
temporal integrity has already been established and hashed, so nothing here
can reach a model, a training window, or a future observation even by mistake.
``tests/test_econ.py::TestBoundary`` asserts the import graph.

THE ALIGNMENT — the leakage-critical line in this file
======================================================
A stored row is already ``(forecast issued at origin, realization at
origin+h)``: ``forecast_var`` is what the model predicted *at* ``origin_index``
and ``realized_return`` is the return at ``target_index = origin_index + h``,
which the splitter guarantees is strictly later. So within one row the position
and the return it earns are correctly staggered by construction, and the whole
backtest is one element-wise product:

    gross_t = position(forecast_var_t) * simple_return(realized_return_t)

The forecast issued AT t sizes the position held INTO t+1 — never the other way
round. Reversing it (sizing today's position with tomorrow's forecast) is the
classic vol-targeting look-ahead, and it flatters Sharpe precisely because a
volatility forecast is informative. ``tests/test_econ.py::TestAlignment``
shifts the realized-return column by one row and asserts the Sharpe moves
materially, so the alignment is pinned by a test that *can* fail rather than by
this paragraph.

Conventions, stated because a Sharpe ratio means nothing without them
====================================================================
- **Returns.** ``realized_return`` is a *log* return (``volbench.data``'s
  ``log_returns``). Position sizing uses it as-is — the forecast variance is a
  variance of that same log return, so the ratio is in consistent units — but
  the P&L converts it to a simple return with ``expm1`` before levering, so
  that the equity curve compounds the way capital actually does and the maximum
  drawdown is a drawdown of capital.
- **Annualization** is per asset, and it is a calendar fact, not a preference:
  equities trade ~252 days a year, crypto trades 365. ``periods_per_year``
  defaults from the asset id (:func:`periods_per_year_for`) and can be
  overridden explicitly. It scales the vol target, the annualized return, the
  annualized vol and the Sharpe, so a crypto and an equity Sharpe computed here
  are comparable.
- **The vol target is stated annualized** (``annual_target_vol=0.10`` — a 10%
  target) because that is how it is always quoted, and converted to the daily
  units everything else in volbench uses exactly once, here:
  ``daily_target = annual_target_vol / sqrt(periods_per_year)``. So one target
  means the same risk on a 252-day and a 365-day calendar.
- **Costs** are ``cost_bps`` basis points of the turnover
  ``|position_t - position_{t-1}|``, charged on the day the position moves.
  The default is **10 bps**, which is docs/research_design.md's "Sharpe net of
  10 bps per rebalance". The first day is charged against a flat book
  (``position_{-1} = 0``), because getting into the position costs money.
- **Risk-free rate** defaults to 0, so the reported Sharpe is a raw-return
  Sharpe. Stated rather than hidden: with ``leverage_cap > 1`` the strategy
  borrows, and no financing spread beyond the transaction costs is modelled.
  Pass ``risk_free_annual`` to net one out.
- **An unusable forecast means no position.** A row whose ``forecast_var`` is
  NaN or non-positive — the evaluator's ``missing_reason`` rows — sizes to
  zero rather than being dropped. Dropping would splice two non-adjacent days
  into one turnover step and quietly under-charge costs; sizing to zero is what
  a desk with no forecast would actually do, and ``n_flat`` reports how often
  it happened.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "CRYPTO_ASSETS",
    "CRYPTO_PERIODS_PER_YEAR",
    "DEFAULT_COST_BPS",
    "DEFAULT_LEVERAGE_CAP",
    "DEFAULT_TARGET_VOL",
    "EQUITY_PERIODS_PER_YEAR",
    "REQUIRED_COLUMNS",
    "VolTargetBacktest",
    "periods_per_year_for",
    "volatility_target_backtest",
]

#: Trading days per year for an exchange-listed equity series.
EQUITY_PERIODS_PER_YEAR: Final = 252.0
#: Crypto trades every calendar day, including weekends and holidays.
CRYPTO_PERIODS_PER_YEAR: Final = 365.0

#: The panel's 24/7 assets. Held here rather than imported from
#: ``volbench.data.panel`` so this module depends on no data-ingestion code —
#: and pinned against ``CRYPTO_PANEL`` by ``tests/test_econ.py``, the same
#: two-lists-one-test discipline D-024 settled on, so the copy cannot drift.
CRYPTO_ASSETS: Final = frozenset({"BTC-USD", "ETH-USD"})

#: docs/research_design.md: "Sharpe net of 10 bps per rebalance".
DEFAULT_COST_BPS: Final = 10.0
#: A 10% annualized volatility target, the usual textbook figure.
DEFAULT_TARGET_VOL: Final = 0.10
#: Positions are capped rather than unbounded: as the forecast vol goes to
#: zero the unconstrained position goes to infinity, and a backtest whose
#: headline number is decided by its three calmest days is not a result.
DEFAULT_LEVERAGE_CAP: Final = 2.0

#: What a stored results frame must carry for this backtest to be computable.
REQUIRED_COLUMNS: Final = (
    "config_hash",
    "asset",
    "origin_index",
    "horizon",
    "forecast_var",
    "realized_return",
)


def periods_per_year_for(asset: str) -> float:
    """Annualization factor for ``asset`` — 365 for crypto, 252 otherwise.

    A calendar fact about the instrument, so it belongs to the asset and not to
    the caller's preference. Override it with the ``periods_per_year`` argument
    when a series really does have a different calendar.
    """
    return CRYPTO_PERIODS_PER_YEAR if asset in CRYPTO_ASSETS else EQUITY_PERIODS_PER_YEAR


@dataclass(frozen=True, eq=False)
class VolTargetBacktest:
    """One cell's volatility-targeting backtest. ``eq=False``: numpy fields.

    Every headline number is net of costs unless its name says ``gross``.
    ``gross_sharpe`` is reported beside ``sharpe`` on purpose: the difference
    between them is what the transaction-cost assumption is worth, and a
    strategy whose edge is entirely inside 10 bps should be visibly so.
    """

    asset: str
    model: str
    config_hash: str
    horizon: int
    n_periods: int
    #: Rows sized to zero because their forecast was unusable.
    n_flat: int
    periods_per_year: float
    annual_target_vol: float
    leverage_cap: float
    cost_bps: float
    risk_free_annual: float

    annual_return: float
    annual_vol: float
    sharpe: float
    gross_sharpe: float
    max_drawdown: float
    avg_leverage: float
    avg_daily_turnover: float
    annual_turnover: float
    annual_cost_drag: float
    #: True if the levered book was wiped out (equity reached zero).
    ruined: bool

    #: Per-period series, in origin order, for inspection and plotting.
    positions: NDArray[np.float64]
    net_returns: NDArray[np.float64]
    equity: NDArray[np.float64]

    def as_dict(self) -> dict[str, Any]:
        """The scalar summary — everything except the per-period series."""
        return {
            "asset": self.asset,
            "model": self.model,
            "config_hash": self.config_hash,
            "horizon": self.horizon,
            "n_periods": self.n_periods,
            "n_flat": self.n_flat,
            "periods_per_year": self.periods_per_year,
            "annual_target_vol": self.annual_target_vol,
            "leverage_cap": self.leverage_cap,
            "cost_bps": self.cost_bps,
            "risk_free_annual": self.risk_free_annual,
            "annual_return": self.annual_return,
            "annual_vol": self.annual_vol,
            "sharpe": self.sharpe,
            "gross_sharpe": self.gross_sharpe,
            "max_drawdown": self.max_drawdown,
            "avg_leverage": self.avg_leverage,
            "avg_daily_turnover": self.avg_daily_turnover,
            "annual_turnover": self.annual_turnover,
            "annual_cost_drag": self.annual_cost_drag,
            "ruined": self.ruined,
        }

    def __str__(self) -> str:
        return (
            f"VolTargetBacktest({self.asset}/{self.model}: Sharpe {self.sharpe:.3f} "
            f"(gross {self.gross_sharpe:.3f}), return {self.annual_return:.2%}, "
            f"vol {self.annual_vol:.2%}, maxDD {self.max_drawdown:.2%}, "
            f"leverage {self.avg_leverage:.2f}, n={self.n_periods})"
        )


def volatility_target_backtest(
    rows: pd.DataFrame,
    *,
    horizon: int = 1,
    annual_target_vol: float = DEFAULT_TARGET_VOL,
    leverage_cap: float = DEFAULT_LEVERAGE_CAP,
    cost_bps: float = DEFAULT_COST_BPS,
    periods_per_year: float | None = None,
    risk_free_annual: float = 0.0,
) -> VolTargetBacktest:
    """Size a position from each stored forecast and score the resulting P&L.

    Parameters
    ----------
    rows:
        Scored rows for **one cell** — one ``config_hash`` and one ``asset``.
        Passing several cells raises rather than pooling them: two models'
        forecasts averaged into one position is not a backtest of either.
    horizon:
        Which horizon's rows to trade. A cell run at ``horizon=5`` holds rows
        for ``h = 1..5``; each is a different strategy (a position held one day
        versus five), so exactly one is selected and it is never inferred.
    annual_target_vol:
        The volatility target, quoted annualized and converted to daily units
        once (module docstring).
    leverage_cap:
        Upper bound on the position. Applied after the ratio, so it binds
        exactly when the forecast vol falls below ``target / cap``.
    cost_bps:
        Basis points charged on turnover ``|position_t - position_{t-1}|``.
        Default 10 (docs/research_design.md).
    periods_per_year:
        Annualization. ``None`` takes it from the asset
        (:func:`periods_per_year_for`): 365 for crypto, 252 for equities.
    risk_free_annual:
        Annualized risk-free rate netted out of the Sharpe numerator. Default
        0; see the module docstring's caveat about financing leverage.

    Returns
    -------
    :class:`VolTargetBacktest`.

    Notes
    -----
    The row is the alignment: ``position`` comes from ``forecast_var`` at
    ``origin_index`` and earns ``realized_return`` at
    ``target_index = origin_index + horizon``. Nothing in this function shifts,
    lags or reindexes anything, which is why there is no place for an
    off-by-one to hide.
    """
    if leverage_cap <= 0.0:
        raise ValueError(f"leverage_cap must be > 0, got {leverage_cap}")
    if annual_target_vol <= 0.0:
        raise ValueError(f"annual_target_vol must be > 0, got {annual_target_vol}")
    if cost_bps < 0.0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    frame = _one_cell(rows, horizon)
    asset = str(frame["asset"].iloc[0])
    ppy = float(periods_per_year) if periods_per_year is not None else periods_per_year_for(asset)
    if ppy <= 0.0:
        raise ValueError(f"periods_per_year must be > 0, got {ppy}")

    forecast_var = frame["forecast_var"].to_numpy(dtype=np.float64)
    log_return = frame["realized_return"].to_numpy(dtype=np.float64)

    daily_target = annual_target_vol / math.sqrt(ppy)
    positions = _positions(forecast_var, daily_target, leverage_cap)
    n_flat = int((positions == 0.0).sum())

    unusable = ~np.isfinite(log_return) & (positions != 0.0)
    if unusable.any():
        first = int(np.flatnonzero(unusable)[0])
        raise ValueError(
            "realized_return must be finite wherever a position is held; first offending "
            f"row is origin_index {int(frame['origin_index'].iloc[first])}"
        )

    # Log -> simple before levering: a position earns the instrument's simple
    # return, and only simple returns compound into an equity curve whose
    # drawdown is a drawdown of capital.
    simple_return = np.where(np.isfinite(log_return), np.expm1(log_return), 0.0)
    gross = positions * simple_return

    # Turnover against a flat book on day one: entering the position is a real
    # trade and a real cost, not a free initial condition.
    turnover = np.abs(np.diff(positions, prepend=0.0))
    costs = turnover * (cost_bps / 10_000.0)
    net = gross - costs

    equity, ruined = _equity_curve(net)
    n = net.size
    rf_daily = risk_free_annual / ppy

    return VolTargetBacktest(
        asset=asset,
        model=str(frame["model"].iloc[0]) if "model" in frame.columns else "",
        config_hash=str(frame["config_hash"].iloc[0]),
        horizon=horizon,
        n_periods=n,
        n_flat=n_flat,
        periods_per_year=ppy,
        annual_target_vol=annual_target_vol,
        leverage_cap=leverage_cap,
        cost_bps=cost_bps,
        risk_free_annual=risk_free_annual,
        annual_return=_annualized_return(equity, n, ppy),
        annual_vol=_annualized_vol(net, ppy),
        sharpe=_sharpe(net - rf_daily, ppy),
        gross_sharpe=_sharpe(gross - rf_daily, ppy),
        max_drawdown=_max_drawdown(equity),
        avg_leverage=float(np.mean(positions)),
        avg_daily_turnover=float(np.mean(turnover)),
        annual_turnover=float(np.mean(turnover)) * ppy,
        annual_cost_drag=float(np.mean(costs)) * ppy,
        ruined=ruined,
        positions=positions,
        net_returns=net,
        equity=equity,
    )


# --------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------


def _one_cell(rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The rows of exactly one cell at one horizon, in origin order.

    Every check here is about making the element-wise product downstream mean
    what it says: one model, one asset, one horizon, one row per origin, in
    time order.
    """
    if not isinstance(rows, pd.DataFrame):
        raise TypeError(f"rows must be a DataFrame, got {type(rows).__name__}")
    missing = [column for column in REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(
            f"results frame is missing columns this backtest needs: {missing}. "
            "Pass rows as stored by ResultsStore (volbench.runner.read_grid_results)."
        )
    hashes = sorted(set(rows["config_hash"].astype(str)))
    if len(hashes) != 1:
        raise ValueError(
            f"expected rows for exactly one cell, got {len(hashes)} config hashes "
            f"({hashes[:3]}...). Backtest one model at a time: averaging two models' "
            "forecasts into one position is a backtest of neither."
        )
    assets = sorted(set(rows["asset"].astype(str)))
    if len(assets) != 1:
        raise ValueError(f"expected rows for exactly one asset, got {assets}")

    frame = rows[rows["horizon"].astype(int) == horizon]
    if frame.empty:
        available = sorted(set(rows["horizon"].astype(int)))
        raise ValueError(f"no rows at horizon {horizon}; this cell has horizons {available}")
    frame = frame.sort_values("origin_index", kind="stable").reset_index(drop=True)
    origins = frame["origin_index"].to_numpy(dtype=np.int64)
    if origins.size > 1 and not bool(np.all(np.diff(origins) > 0)):
        raise ValueError(
            "origin_index must be strictly increasing after selecting one horizon; "
            "duplicate origins would double-count a day's P&L"
        )
    return frame


def _positions(
    forecast_var: NDArray[np.float64], daily_target: float, cap: float
) -> NDArray[np.float64]:
    """``min(target / forecast_vol, cap)``, and 0 where there is no forecast.

    Computed with an explicit mask rather than by dividing and cleaning up
    afterwards: ``target / sqrt(nan)`` is NaN and ``target / sqrt(0)`` is
    ``inf``, and both would survive a naive ``np.minimum`` against the cap as
    NaN and ``cap`` respectively — one poisoning every downstream mean, the
    other silently taking maximum leverage on the strength of a forecast that
    does not exist.
    """
    usable = np.isfinite(forecast_var) & (forecast_var > 0.0)
    positions = np.zeros_like(forecast_var, dtype=np.float64)
    positions[usable] = np.minimum(daily_target / np.sqrt(forecast_var[usable]), cap)
    return positions


def _equity_curve(net: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]:
    """Compounded wealth from 1.0, floored at zero if the book is wiped out.

    A levered position can lose more than the capital behind it. Reporting a
    negative equity curve — and a drawdown computed from it — would be
    arithmetic, not a result, so the curve stops at zero and the caller is told
    both in ``ruined`` and by a warning.
    """
    growth = np.asarray(1.0 + net, dtype=np.float64)
    ruined = bool((growth <= 0.0).any())
    if not ruined:
        return np.cumprod(growth, dtype=np.float64), False
    first = int(np.flatnonzero(growth <= 0.0)[0])
    warnings.warn(
        f"the levered position was wiped out at period {first}: a return of "
        f"{net[first]:.4f} against a position of that size takes equity to zero. "
        "Everything after it is reported as zero, not as a recovery.",
        RuntimeWarning,
        stacklevel=3,
    )
    curve = np.zeros_like(net, dtype=np.float64)
    curve[:first] = np.cumprod(growth[:first], dtype=np.float64)
    return curve, True


def _annualized_return(equity: NDArray[np.float64], n: int, ppy: float) -> float:
    """Geometric: what constant annual rate would produce this final wealth."""
    if n == 0:
        return math.nan
    final = float(equity[-1])
    if final <= 0.0:
        return -1.0
    return float(final ** (ppy / n) - 1.0)


def _annualized_vol(net: NDArray[np.float64], ppy: float) -> float:
    if net.size < 2:
        return math.nan
    return float(np.std(net, ddof=1) * math.sqrt(ppy))


def _sharpe(excess: NDArray[np.float64], ppy: float) -> float:
    """Annualized mean over annualized standard deviation of excess returns.

    ``nan`` when the dispersion is not meaningfully positive. The guard is
    *scale-relative*, not ``sd > 0``, because a numerically constant series
    does not have ``sd == 0`` — it has ``sd`` at the float64 noise floor, and
    dividing by that gives a Sharpe of 1e16 that changes between runs. A real
    return series never comes near the threshold (even one with a single
    non-zero observation has ``sd/max|r| ~ 1/sqrt(n)``), so this can only fire
    on a degenerate input, where the honest answer is "undefined".
    """
    if excess.size < 2:
        return math.nan
    sd = float(np.std(excess, ddof=1))
    scale = float(np.max(np.abs(excess)))
    if sd <= 0.0 or (scale > 0.0 and sd < 1e-12 * scale):
        return math.nan
    return float(np.mean(excess) / sd * math.sqrt(ppy))


def _max_drawdown(equity: NDArray[np.float64]) -> float:
    """Largest peak-to-trough fall of the equity curve, as a positive fraction."""
    if equity.size == 0:
        return math.nan
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))
    curve = np.concatenate([[1.0], equity])
    return float(np.max(1.0 - curve / peak))
