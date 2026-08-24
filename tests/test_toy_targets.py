"""One scoring target per evaluation cell — never per model (M2 review, item 1).

Supersedes the short-lived per-model wiring: with per-model targets the QLIKE
column compared models against *different* proxies, which is not a comparison.
The pins:

- every model in a run is scored against the same target, the benchmark-level
  ``SCORING_TARGET`` (overnight + Rogers-Satchell close-to-close estimator);
- Parkinson survives as a *labeled* robustness arm behind the ``target`` flag,
  never as a silent default;
- the flag is an evaluation knob only: forecasts do not depend on the proxy,
  so switching the target moves QLIKE and the proxy columns and NOTHING else —
  and HAR's fit input stays the close-to-close series under either target,
  because what a model forecasts is a modelling contract, not an evaluation
  setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volbench.benchmarks.toy import (
    ROBUSTNESS_TARGET,
    SCORING_TARGET,
    load_series,
    models,
    run_toy_benchmark,
)

FORECAST_COLUMNS = ["forecast_mean", "forecast_var", "crps", "log_score", "realized_return"]


@pytest.fixture(scope="module")
def runs() -> dict[str, pd.DataFrame]:
    return {
        target: run_toy_benchmark(out_dir=None, target=target).results
        for target in (SCORING_TARGET, ROBUSTNESS_TARGET)
    }


def test_every_cell_in_a_run_shares_the_one_scoring_target(
    runs: dict[str, pd.DataFrame],
) -> None:
    for target, frame in runs.items():
        assert set(frame["proxy_name"]) == {target}
    assert SCORING_TARGET == "overnight_plus_range"  # the default is the close-to-close target


def test_no_model_carries_its_own_target() -> None:
    assert not hasattr(models()[0], "target")


def test_switching_the_target_moves_qlike_and_nothing_else(
    runs: dict[str, pd.DataFrame],
) -> None:
    """The proxy never reaches a model, so the robustness arm re-scores the
    *same* forecasts. Includes HAR: its fit input is pinned to the
    close-to-close series under both targets."""
    default = runs[SCORING_TARGET].sort_values(["label", "origin_index"]).reset_index(drop=True)
    robust = runs[ROBUSTNESS_TARGET].sort_values(["label", "origin_index"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(default[FORECAST_COLUMNS], robust[FORECAST_COLUMNS])
    for level_tag in ("0p01", "0p025", "0p05"):
        pd.testing.assert_series_equal(default[f"hit_{level_tag}"], robust[f"hit_{level_tag}"])
        pd.testing.assert_series_equal(
            default[f"pinball_{level_tag}"], robust[f"pinball_{level_tag}"]
        )
    # QLIKE and the proxy columns are what the flag exists to change.
    assert not np.allclose(default["qlike"], robust["qlike"])
    assert not np.allclose(default["proxy_var"], robust["proxy_var"])
    # And the experiment identities differ: the target is part of the hash.
    assert set(default["config_hash"]) != set(robust["config_hash"])


def test_hars_fit_input_is_the_close_to_close_series_under_either_target() -> None:
    toy = load_series()
    har = next(e for e in models() if e.label == "har")
    for target in (SCORING_TARGET, ROBUSTNESS_TARGET):
        _, proxy, fit_series = toy.inputs_for(har, target=target)
        assert fit_series is not None
        pd.testing.assert_series_equal(fit_series, toy.targets[SCORING_TARGET])
    # while the proxy tracks the flag
    _, proxy, _ = toy.inputs_for(har, target=ROBUSTNESS_TARGET)
    pd.testing.assert_series_equal(proxy, toy.targets[ROBUSTNESS_TARGET])


def test_return_fed_models_take_no_fit_series() -> None:
    toy = load_series()
    for entry in models():
        if entry.label == "har":
            continue
        _, _, fit_series = toy.inputs_for(entry)
        assert fit_series is None


def test_an_unknown_target_is_rejected() -> None:
    toy = load_series()
    with pytest.raises(KeyError, match="unknown target"):
        toy.inputs_for(models()[0], target="yang_zhang")
    with pytest.raises(KeyError, match="unknown target"):
        run_toy_benchmark(out_dir=None, target="squared_return")


def test_the_default_target_carries_the_overnight_variance_parkinson_omits() -> None:
    """Why the default is the close-to-close estimator at all: on the committed
    fixture it is larger than Parkinson by about the overnight share."""
    toy = load_series()
    park = toy.targets[ROBUSTNESS_TARGET].to_numpy()
    opr = toy.targets[SCORING_TARGET].to_numpy()
    assert 0.04 < float(np.mean(opr) / np.mean(park)) - 1.0 < 0.20
