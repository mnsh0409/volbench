"""Generate the synthetic daily OHLC series the M1 toy benchmark runs on.

WHY SYNTHETIC, AND NOT A REAL ASSET
-----------------------------------
The M1 brief asks for "one real asset (or a committed fixture if the network
is unavailable)". The network is *not* the binding constraint here — the
licence is, and that is a deliberate finding rather than a shortcut:

- **Stooq** (the D-004 index panel): redistribution is an explicit NO
  (ToS §5.3, read 2026-08-23; §6.1 restricts the S&P Dow Jones series further
  to personal, non-commercial use). Automated download is additionally gated
  behind a JS proof-of-work anti-bot challenge, re-verified on 2026-08-23 and
  again at M1 integration — the endpoint answers, but with the challenge page,
  not CSV. volbench does not attempt to bypass it.
- **Binance** (the crypto RV arm): the archive is reachable, but whether the
  *derived* daily-RV series may be redistributed is unconfirmed pending a
  human read of the full Terms of Use.

CLAUDE.md forbids vendoring data that is not clearly redistributable, and the
data stream's own rule forbids tests that reach the network. Committing a real
index series to make a smoke test convenient would break the first rule to
satisfy the second. So the toy benchmark runs on a series with no licence
attached at all, generated here.

WHAT IT IS
----------
A GARCH(1,1) return process with realistic daily-equity persistence, wrapped
in a Brownian-bridge intraday path so the high/low are consistent with each
day's own close-to-close move. That matters: the benchmark feeds the
Parkinson range proxy to HAR-RV, and a fixture with made-up highs and lows
would produce a range proxy unrelated to the returns being scored, making the
QLIKE column meaningless as a smoke signal.

The numbers are a plausibility harness, NOT evidence about any model. No
result from this fixture belongs in the paper.

Rebuild (byte-identical, fixed seed)::

    uv run python -m volbench.benchmarks.make_toy_asset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

__all__ = ["DEFAULT_PATH", "simulate_ohlc", "write_toy_asset"]

DEFAULT_PATH = Path(__file__).parent / "data" / "toy_asset_daily.csv"

#: 701 daily bars -> 700 usable returns after the leading gap is trimmed ->
#: exactly 200 rolling origins at window=500, horizon=1, step=1.
N_DAYS = 701
SEED = 20260823
START_PRICE = 4000.0
START_DATE = "2019-01-02"

# GARCH(1,1) parameters: persistence alpha+beta = 0.98 (typical for a daily
# equity index), unconditional daily sigma = sqrt(2.4e-6 / 0.02) ~ 1.1%.
OMEGA, ALPHA, BETA = 2.4e-6, 0.06, 0.92

#: Intraday steps per day for the high/low path: one per minute of a 6.5-hour
#: US equity session. Resolution matters more than it looks. Parkinson and
#: Garman-Klass assume a continuously-observed path, so the high and low of a
#: coarsely-sampled one sit inside the true range and the proxy is biased low
#: — at 13 steps it came out ~30% below the return variance, which swamped the
#: real effects the toy table is meant to show. At 390 the bias is ~4%.
INTRADAY_STEPS = 390
#: Overnight gap volatility as a fraction of the day's sigma.
GAP_FRACTION = 0.3
#: Price decimals. Deliberately fine enough that no day can round to high ==
#: low: a zero range would make the Parkinson proxy 0, and HAR-RV takes logs
#: of it. See docs/M1_REPORT.md risk 2 — that fragility is real on rounded
#: real-world data, and this fixture is not the place to trip over it.
PRICE_DECIMALS = 4


def simulate_ohlc(
    n_days: int = N_DAYS, seed: int = SEED, start_price: float = START_PRICE
) -> pd.DataFrame:
    """Simulate ``n_days`` of daily OHLC from a GARCH(1,1) return process.

    Every bar satisfies ``low <= min(open, close) <= max(open, close) <= high``
    and ``high > low`` by construction, because the high and low are read off
    an actual simulated intraday path rather than drawn independently.
    """
    rng = np.random.default_rng(seed)

    z = rng.standard_normal(n_days)
    sigma2 = np.empty(n_days, dtype=np.float64)
    returns = np.empty(n_days, dtype=np.float64)
    sigma2[0] = OMEGA / (1.0 - ALPHA - BETA)  # unconditional variance
    returns[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n_days):
        sigma2[t] = OMEGA + ALPHA * returns[t - 1] ** 2 + BETA * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * z[t]

    sigma = np.sqrt(sigma2)
    gap = GAP_FRACTION * sigma * rng.standard_normal(n_days)
    intraday = returns - gap  # so gap + intraday is exactly the day's return

    close_prev = start_price
    rows: list[tuple[float, float, float, float]] = []
    for t in range(n_days):
        open_t = close_prev * np.exp(gap[t])
        path = _bridge(rng, steps=INTRADAY_STEPS, endpoint=intraday[t], vol=sigma[t])
        prices = open_t * np.exp(path)
        close_t = float(prices[-1])
        rows.append((open_t, float(prices.max()), float(prices.min()), close_t))
        close_prev = close_t

    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"]).round(PRICE_DECIMALS)
    frame.insert(0, "date", pd.date_range(START_DATE, periods=n_days, freq="B", tz="UTC"))
    return frame


def _bridge(
    rng: np.random.Generator, *, steps: int, endpoint: float, vol: float
) -> NDArray[np.float64]:
    """Brownian bridge from 0 to ``endpoint`` in ``steps`` steps, including both ends.

    The high and low of the day are the running max/min of this path, so they
    are consistent with the close the day actually printed.
    """
    increments = vol / np.sqrt(steps) * rng.standard_normal(steps)
    walk = np.concatenate([[0.0], np.cumsum(increments)])
    frac = np.arange(steps + 1, dtype=np.float64) / steps
    bridge: NDArray[np.float64] = walk - frac * walk[-1] + frac * endpoint
    return bridge


def write_toy_asset(path: Path = DEFAULT_PATH) -> Path:
    """Write the fixture to ``path`` and return it."""
    frame = simulate_ohlc()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    written = write_toy_asset(args.out)
    frame = pd.read_csv(written)
    print(f"wrote {len(frame)} daily bars to {written}")


if __name__ == "__main__":
    main()
