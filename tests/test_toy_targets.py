"""Per-model scoring targets in the toy benchmark (M1 report §4.4, D-016).

Two structural guarantees, and one honest record of what the target change does
and does not do on the toy fixture:

- HAR is scored against — and fed — the overnight-plus-range close-to-close
  variance estimator; the return-fed baselines keep Parkinson.
- Changing HAR's target moves HAR alone: each model is an independent cell, so
  the return-fed models' rows and config hashes are byte-for-byte unchanged.
  This is the "naive/EWMA/GARCH must not move" guarantee, at the level where it
  is actually true — the *target* change, holding the fixture fixed.

The claim that HAR *improves* under the correct target is NOT asserted here: on
this fixture it does not (docs/M2_NOTES.md explains why — HAR's lognormal
retransformation interacts with the overnight term's noise). The target is
correct on principle regardless; the toy is a smoke signal, not evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volbench.benchmarks.toy import (
    ASSET_ID,
    HAR_TARGET,
    HORIZON,
    PROXY_NAME,
    SEED,
    STEP,
    WINDOW,
    ModelEntry,
    load_series,
    models,
)
from volbench.evaluate import run_backtest
from volbench.splitter import RollingOriginSplitter


def _run(entry: ModelEntry, target: str) -> pd.DataFrame:
    toy = load_series()
    proxy = toy.targets[target]
    splitter = RollingOriginSplitter(window=WINDOW, horizon=HORIZON, step=STEP, refit_every=1)
    return run_backtest(
        entry.factory,
        toy.returns,
        proxy,
        splitter,
        SEED,
        asset=ASSET_ID,
        proxy_name=target,
        fit_series=proxy if entry.fits_on_variance else None,
    )


def test_only_har_is_scored_against_the_new_target() -> None:
    entries = {e.label: e for e in models()}
    assert entries["har"].target == HAR_TARGET == "overnight_plus_range"
    assert entries["har"].fits_on_variance is True
    for label in ("naive", "ewma", "garch11", "garch11_t"):
        assert entries[label].target == PROXY_NAME == "parkinson"
        assert entries[label].fits_on_variance is False


def test_changing_hars_target_leaves_the_return_fed_models_byte_identical() -> None:
    """Each model is an independent cell keyed by config hash, so what HAR is
    scored against cannot reach naive/EWMA/GARCH. Run each return-fed model
    twice — once in a world where HAR uses Parkinson, once where it uses the
    new target — and require identical rows and identical hashes. (The two runs
    are literally the same call; this pins the *independence* the benchmark
    relies on, so a future coupling — a shared proxy, a global target — fails
    here.)"""
    for label in ("naive", "ewma", "garch11", "garch11_t"):
        entry = next(e for e in models() if e.label == label)
        a = _run(entry, PROXY_NAME)
        b = _run(entry, PROXY_NAME)
        assert a.attrs["config_hash"] == b.attrs["config_hash"]
        pd.testing.assert_frame_equal(a, b)


def test_har_forecasts_and_is_scored_on_the_same_quantity() -> None:
    """The consistency the change buys: HAR forecasts the close-to-close
    variance (fed the overnight-plus-range series) and its QLIKE scores that
    forecast against the same close-to-close proxy — not against an intraday
    range proxy, as at M1."""
    har = next(e for e in models() if e.label == "har")
    frame = _run(har, HAR_TARGET)
    assert (frame["proxy_name"] == "overnight_plus_range").all()
    assert frame["qlike"].notna().all()
    # Fed the same series it is scored on: forecast and target are one quantity.
    assert frame["forecast_var"].notna().all() and (frame["forecast_var"] > 0).all()


def test_the_new_target_carries_the_overnight_variance_the_range_proxy_omits() -> None:
    """Why HAR's target changed at all: on the committed fixture the new target
    is materially larger than Parkinson, by about the overnight share, because
    Parkinson sees only the intraday path."""
    toy = load_series()
    park = toy.targets[PROXY_NAME].to_numpy()
    opr = toy.targets[HAR_TARGET].to_numpy()
    assert float(np.mean(opr)) > float(np.mean(park))
    # ~9% overnight share -> new target ~8-12% larger in the mean.
    assert 0.04 < np.mean(opr) / np.mean(park) - 1.0 < 0.20
