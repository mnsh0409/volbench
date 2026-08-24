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
A GARCH(1,1) *daily* variance process with realistic equity persistence,
realized each day as two independent pieces — an overnight jump from the
previous close to the open, and an intraday random walk from the open to the
close — so the high and low are read off an actual path consistent with the
close the day printed. That matters: the benchmark feeds a range-based
variance estimator to HAR-RV, and a fixture with made-up highs and lows
would produce a target unrelated to the returns being scored, making the
QLIKE column meaningless as a smoke signal.

THE COMPONENT MODEL (changed at M2, docs/M1_REPORT.md §4.4)
-------------------------------------------------------------
Day ``t`` has conditional variance ``sigma2[t]`` from the GARCH recursion on
close-to-close returns. Its overnight jump is ``N(0, OVERNIGHT_SHARE *
sigma2[t])`` and its intraday path is a random walk with total variance
``(1 - OVERNIGHT_SHARE) * sigma2[t]``, drawn independently, so

    var(close-to-close) = var(overnight) + var(open-to-close) = sigma2[t]

exactly, and the fixture *knows* all three. The M1 generator drew the
close-to-close return first and carved a gap out of it, which made the
intraday variance ``1.09 sigma2`` and the gap negatively correlated with the
rest of the day; a range estimator then recovered the *whole* daily variance
and an overnight-plus-range estimator over-shot it. That is not how markets
work, and it made the fixture useless for judging which estimator targets
the close-to-close variance. Independent components are the standard model
behind Yang & Zhang (2000) and are what the validation in
tests/test_target_estimators.py relies on. The regeneration moved every
number in the toy benchmark; docs/M2_NOTES.md records old vs new.

The numbers are a plausibility harness, NOT evidence about any model. No
result from this fixture belongs in the paper.

Rebuild (byte-identical, fixed seed)::

    uv run python -m volbench.benchmarks.make_toy_asset
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ["DEFAULT_PATH", "OVERNIGHT_SHARE", "simulate_ohlc", "write_toy_asset"]

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

#: Intraday steps per day for the high/low path. Range estimators (Parkinson,
#: Rogers-Satchell) assume a continuously-observed path, so the high and low
#: of a coarsely-sampled one sit inside the true range and the estimator is
#: biased low; that discretization bias must sit well below the ~9% overnight
#: share, or it would masquerade as the structural effect the M2 target change
#: is meant to demonstrate. Measured against the fixture's known variance:
#: at 390 steps Rogers-Satchell is ~8% low (comparable to the overnight
#: share — useless for the comparison), at 5000 steps ~0.8% (negligible), so
#: Parkinson's residual bias is then essentially the overnight gap alone.
#: Generation stays ~0.05s, and `make reproduce` regenerates from this.
INTRADAY_STEPS = 5000
#: Share of each day's variance carried by the overnight jump (close-to-open),
#: the rest being the open-to-close random walk. 9% is a plausible equity-
#: index figure; the fixture's point is that the share is *known*, so an
#: estimator's bias against the true close-to-close variance can be measured.
OVERNIGHT_SHARE = 0.09
#: Price decimals. Deliberately fine enough that no day can round to high ==
#: low: a zero range would make the Parkinson proxy 0, and HAR-RV takes logs
#: of it. See docs/M1_REPORT.md risk 2 — that fragility is real on rounded
#: real-world data, and this fixture is not the place to trip over it.
PRICE_DECIMALS = 4


def simulate_ohlc(
    n_days: int = N_DAYS,
    seed: int = SEED,
    start_price: float = START_PRICE,
    *,
    intraday_steps: int = INTRADAY_STEPS,
    overnight_share: float = OVERNIGHT_SHARE,
) -> pd.DataFrame:
    """Simulate ``n_days`` of daily OHLC from a GARCH(1,1) variance process.

    Returns ``date, open, high, low, close`` plus ``true_variance`` — the
    conditional close-to-close variance ``sigma2[t]`` the bar was drawn with,
    so tests can measure an estimator's bias against the truth. The data
    layer's loaders keep only OHLCV columns, so the extra column never
    reaches a model.

    Every bar satisfies ``low <= min(open, close) <= max(open, close) <= high``
    and ``high > low`` by construction, because the high and low are read off
    an actual simulated intraday path rather than drawn independently.
    ``intraday_steps`` and ``overnight_share`` exist so the validation tests
    can vary the path resolution and the component split; the committed
    fixture uses the module defaults.
    """
    if not 0.0 <= overnight_share < 1.0:
        raise ValueError("overnight_share must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    unconditional = OMEGA / (1.0 - ALPHA - BETA)

    sigma2 = np.empty(n_days, dtype=np.float64)
    returns = np.empty(n_days, dtype=np.float64)
    close_prev = start_price
    rows: list[tuple[float, float, float, float]] = []
    for t in range(n_days):
        sigma2[t] = (
            unconditional
            if t == 0
            else OMEGA + ALPHA * returns[t - 1] ** 2 + BETA * sigma2[t - 1]
        )
        # Two independent pieces whose variances add up to sigma2[t].
        gap = math.sqrt(overnight_share * sigma2[t]) * rng.standard_normal()
        path = _intraday_path(
            rng, steps=intraday_steps, vol=math.sqrt((1.0 - overnight_share) * sigma2[t])
        )
        returns[t] = gap + float(path[-1])

        open_t = close_prev * math.exp(gap)
        prices = open_t * np.exp(path)
        close_t = float(prices[-1])
        rows.append((open_t, float(prices.max()), float(prices.min()), close_t))
        close_prev = close_t

    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"]).round(PRICE_DECIMALS)
    frame.insert(0, "date", pd.date_range(START_DATE, periods=n_days, freq="B", tz="UTC"))
    frame["true_variance"] = sigma2
    return frame


def _intraday_path(
    rng: np.random.Generator, *, steps: int, vol: float
) -> NDArray[np.float64]:
    """Random walk of ``steps`` increments from 0, total variance ``vol**2``.

    Its endpoint is the day's open-to-close log return and its running
    max/min are the high and low, so all four prices come from one path.
    """
    increments = vol / np.sqrt(steps) * rng.standard_normal(steps)
    walk: NDArray[np.float64] = np.concatenate([[0.0], np.cumsum(increments)])
    return walk


def write_toy_asset(path: Path = DEFAULT_PATH) -> Path:
    """Write the fixture to ``path`` and return it.

    ``true_variance`` is written with enough digits to round-trip a float64,
    so the committed file carries the exact truth the bars were drawn with.
    """
    frame = simulate_ohlc()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.17g")
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
