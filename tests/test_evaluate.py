"""Rolling-origin backtesting: scoring, refit cadence, leakage, determinism.

Two of these are load-bearing for the whole repo and should be the first
things checked when anything downstream looks odd:

- ``test_future_data_cannot_touch_earlier_forecasts`` — the leakage canary
  demanded by ``.claude/skills/leakage-check``.
- ``test_two_identical_runs_produce_identical_parquet`` — the determinism
  canary behind CLAUDE.md rule 3 and `make reproduce`.

Every train/test index in this file comes from ``RollingOriginSplitter``;
there is no hand-rolled slicing here either.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from volbench.dist import Distribution, Normal
from volbench.evaluate import (
    DEFAULT_LEVELS,
    ForecastModel,
    forecast_moments,
    run_backtest,
)
from volbench.execute import SerialExecutor
from volbench.results import ResultsStore
from volbench.splitter import RollingOriginSplitter

T = TypeVar("T")
R = TypeVar("R")

SIGMA = 0.012  # daily return sd, ~19% annualized — a realistic equity index


# --------------------------------------------------------------------------
# test doubles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _NormalFit:
    mu: float
    sigma: float

    def predict(self, h: int) -> Distribution:
        return Distribution.from_normal(self.mu, self.sigma)


@dataclass(frozen=True)
class OracleNormal:
    """Forecasts a fixed N(mu, sigma^2), ignoring the data.

    Lets a test state "this model is exactly right" / "this one is wrong by a
    factor of three" without any estimation noise in the way.
    """

    sigma: float
    mu: float = 0.0
    label: str = "oracle"

    @property
    def name(self) -> str:
        return self.label

    def spec(self) -> dict[str, Any]:
        return {"kind": "oracle", "mu": self.mu, "sigma": self.sigma}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _NormalFit:
        return _NormalFit(self.mu, self.sigma)


class SpyModel:
    """Estimates sigma from the training window and records every fit."""

    def __init__(self, label: str = "spy") -> None:
        self.label = label
        self.fit_calls = 0
        self.predict_calls = 0
        self.fit_windows: list[NDArray[np.float64]] = []

    @property
    def name(self) -> str:
        return self.label

    def spec(self) -> dict[str, Any]:
        return {"kind": "spy"}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _SpyFit:
        self.fit_calls += 1
        self.fit_windows.append(np.asarray(train, dtype=np.float64).copy())
        return _SpyFit(self, float(np.std(train)))


@dataclass(frozen=True, eq=False)
class _SpyFit:
    owner: SpyModel
    sigma: float

    def predict(self, h: int) -> Distribution:
        self.owner.predict_calls += 1
        return Distribution.from_normal(0.0, max(self.sigma, 1e-12))


class UpdatingModel:
    """A model that re-conditions between refits without re-estimating.

    Stands in for GARCH: parameters come from the scheduled refit, the
    conditioning observation is refreshed every origin.
    """

    def __init__(self) -> None:
        self.fit_calls = 0
        self.update_calls = 0

    @property
    def name(self) -> str:
        return "updating"

    def spec(self) -> dict[str, Any]:
        return {"kind": "updating"}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _UpdatingFit:
        self.fit_calls += 1
        return _UpdatingFit(self, float(np.std(train)), abs(float(train[-1])))


@dataclass(frozen=True, eq=False)
class _UpdatingFit:
    owner: UpdatingModel
    scale: float  # estimated at the last refit, never re-estimated
    last_abs: float  # refreshed every origin

    def predict(self, h: int) -> Distribution:
        return Distribution.from_normal(0.0, max(0.5 * self.scale + 0.5 * self.last_abs, 1e-12))

    def update(self, train: NDArray[np.float64]) -> _UpdatingFit:
        self.owner.update_calls += 1
        return _UpdatingFit(self.owner, self.scale, abs(float(train[-1])))


@dataclass(frozen=True)
class EnsembleModel:
    """Returns a sample-based Distribution, which has no tractable density."""

    n_samples: int = 400

    @property
    def name(self) -> str:
        return "ensemble"

    def spec(self) -> dict[str, Any]:
        return {"kind": "ensemble", "n_samples": self.n_samples}

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> _EnsembleFit:
        return _EnsembleFit(float(np.std(train)), self.n_samples)


@dataclass(frozen=True)
class _EnsembleFit:
    sigma: float
    n_samples: int

    def predict(self, h: int) -> Distribution:
        # Deterministic quasi-sample: no RNG, so the canary stays meaningful.
        grid = (np.arange(self.n_samples) + 0.5) / self.n_samples
        return Distribution.from_samples(
            np.array([Normal(0.0, self.sigma).quantile(float(u)) for u in grid])
        )


class CountingExecutor:
    """Records what the backtest handed to the execution seam."""

    def __init__(self) -> None:
        self.map_calls = 0
        self.n_items = 0

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        materialized = list(items)
        self.map_calls += 1
        self.n_items += len(materialized)
        return [fn(item) for item in materialized]


class ShuffledExecutor:
    """Evaluates items in a permuted order, returns results in item order.

    Honours the Executor contract while proving no block depends on the order
    the others ran in — the property the process and Slurm backends rely on.
    """

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        materialized = list(items)
        order = list(range(len(materialized)))
        random.Random(0).shuffle(order)
        results: dict[int, R] = {i: fn(materialized[i]) for i in order}
        return [results[i] for i in range(len(materialized))]


class OutOfOrderExecutor:
    """Deliberately returns results reversed.

    This *violates* the Executor contract; it is here to prove that
    ``normalize_frame`` makes the stored bytes independent of result order
    anyway, which is what D-011's backend-invariance claim rests on.
    """

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        return [fn(item) for item in items][::-1]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def simulated_panel(
    n: int = 600, sigma: float = SIGMA, seed: int = 20260823
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """i.i.d. N(0, sigma^2) daily returns and their squared-return proxy.

    Squared returns are an unbiased (very noisy) variance proxy, so expected
    QLIKE is minimized at the true variance — which is what makes the
    well-specified/misspecified comparison below a real test.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, sigma, size=n)
    return returns, returns**2


def make_splitter(refit_every: int = 1, horizon: int = 1, step: int = 1) -> RollingOriginSplitter:
    return RollingOriginSplitter(window=250, horizon=horizon, step=step, refit_every=refit_every)


def backtest(
    model: ForecastModel,
    returns: NDArray[np.float64],
    proxy: NDArray[np.float64],
    splitter: RollingOriginSplitter,
    **kwargs: Any,
) -> pd.DataFrame:
    return run_backtest(
        lambda: model,
        returns,
        proxy,
        splitter,
        seed=11,
        asset="SIM",
        proxy_name="squared_return",
        **kwargs,
    )


# --------------------------------------------------------------------------
# scoring behaviour
# --------------------------------------------------------------------------


def test_a_well_specified_model_beats_a_misspecified_one() -> None:
    """The sanity floor for the whole benchmark: if the truth does not win on
    CRPS and QLIKE against data simulated from it, the scoring is wrong."""
    returns, proxy = simulated_panel()
    splitter = make_splitter()

    truth = backtest(OracleNormal(SIGMA, label="truth"), returns, proxy, splitter)
    too_wide = backtest(OracleNormal(3.0 * SIGMA, label="wide"), returns, proxy, splitter)
    too_narrow = backtest(OracleNormal(SIGMA / 3.0, label="narrow"), returns, proxy, splitter)

    for metric in ("crps", "qlike", "log_score"):
        assert truth[metric].mean() < too_wide[metric].mean(), metric
        assert truth[metric].mean() < too_narrow[metric].mean(), metric

    # Pinball at the tail levels agrees; that is the risk-relevant direction.
    for level in DEFAULT_LEVELS:
        column = f"pinball_{level:.10g}".replace(".", "p")
        assert truth[column].mean() < too_wide[column].mean()
        assert truth[column].mean() < too_narrow[column].mean()


def test_var_hit_rate_matches_the_nominal_level_for_a_correct_model() -> None:
    returns, proxy = simulated_panel(n=3000)
    frame = backtest(OracleNormal(SIGMA), returns, proxy, make_splitter())
    for level in DEFAULT_LEVELS:
        rate = frame[f"hit_{level:.10g}".replace(".", "p")].mean()
        assert abs(rate - level) < 0.4 * level, (level, rate)


def test_var_quantile_and_hit_indicator_agree() -> None:
    returns, proxy = simulated_panel(n=400)
    frame = backtest(OracleNormal(SIGMA), returns, proxy, make_splitter())
    for level in DEFAULT_LEVELS:
        tag = f"{level:.10g}".replace(".", "p")
        expected = (frame["realized_return"] < frame[f"var_{tag}"]).astype(float)
        assert frame[f"hit_{tag}"].tolist() == expected.tolist()
        assert (frame[f"var_{tag}"] < 0).all()  # left tail of a zero-mean return


def test_forecast_variance_is_what_qlike_scores() -> None:
    returns, proxy = simulated_panel(n=400)
    frame = backtest(OracleNormal(SIGMA), returns, proxy, make_splitter())
    assert np.allclose(frame["forecast_var"], SIGMA**2)
    assert np.allclose(frame["forecast_mean"], 0.0)
    ratio = frame["proxy_var"] / frame["forecast_var"]
    assert np.allclose(frame["qlike"], ratio - np.log(ratio) - 1.0)


def test_one_row_per_origin_and_horizon_with_indices_from_the_splitter() -> None:
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter(horizon=5, step=3)
    frame = backtest(OracleNormal(SIGMA), returns, proxy, splitter)

    origins = list(splitter.split(returns.size))
    assert len(frame) == splitter.n_splits(returns.size) * splitter.horizon
    expected = sorted((o.origin, h, int(t)) for o in origins for h, t in enumerate(o.test, start=1))
    actual = sorted(
        zip(
            frame["origin_index"].tolist(),
            frame["horizon"].tolist(),
            frame["target_index"].tolist(),
            strict=True,
        )
    )
    assert actual == expected
    # The realized target is read at the splitter's test index, nowhere else.
    assert np.array_equal(
        frame["realized_return"].to_numpy(), returns[frame["target_index"].to_numpy()]
    )


# --------------------------------------------------------------------------
# missingness — recorded, never dropped
# --------------------------------------------------------------------------


def test_nan_proxies_are_recorded_not_dropped() -> None:
    returns, proxy = simulated_panel(n=400)
    proxy = proxy.copy()
    proxy[::7] = np.nan
    splitter = make_splitter()
    frame = backtest(OracleNormal(SIGMA), returns, proxy, splitter)

    assert len(frame) == splitter.n_splits(returns.size)  # nothing dropped
    missing = np.isnan(proxy[frame["target_index"].to_numpy()])
    assert missing.any()
    assert frame.loc[missing, "qlike"].isna().all()
    assert (frame.loc[missing, "missing_reason"] == "proxy_nan").all()
    # Everything not needing the proxy is still scored.
    assert frame["crps"].notna().all()
    assert frame.loc[~missing, "qlike"].notna().all()
    assert (frame.loc[~missing, "missing_reason"] == "").all()


@pytest.mark.parametrize(
    ("bad_value", "reason"),
    [(np.nan, "proxy_nan"), (np.inf, "proxy_not_finite"), (-1e-4, "proxy_nonpositive")],
)
def test_every_unusable_proxy_says_why(bad_value: float, reason: str) -> None:
    returns, proxy = simulated_panel(n=300)
    proxy = proxy.copy()
    proxy[:] = bad_value
    frame = backtest(OracleNormal(SIGMA), returns, proxy, make_splitter())
    assert frame["qlike"].isna().all()
    assert (frame["missing_reason"] == reason).all()


def test_a_nan_target_is_recorded_not_dropped() -> None:
    returns, proxy = simulated_panel(n=400)
    returns = returns.copy()
    splitter = make_splitter()
    # Poison a target the splitter actually forecasts.
    poisoned = int(list(splitter.split(returns.size))[3].test[0])
    returns[poisoned] = np.nan

    frame = backtest(OracleNormal(SIGMA), returns, proxy, splitter)
    row = frame.loc[frame["target_index"] == poisoned].iloc[0]
    assert row["missing_reason"] == "target_nan"
    assert math.isnan(row["crps"])
    assert math.isnan(row["log_score"])
    assert math.isnan(row["hit_0p01"])
    assert not math.isnan(row["var_0p01"])  # the forecast itself is fine
    assert len(frame) == splitter.n_splits(returns.size)


def test_a_distribution_without_a_density_reports_an_undefined_log_score() -> None:
    returns, proxy = simulated_panel(n=400)
    frame = backtest(EnsembleModel(), returns, proxy, make_splitter())
    assert frame["log_score"].isna().all()
    assert (frame["missing_reason"] == "log_score_undefined").all()
    assert frame["crps"].notna().all()
    assert frame["qlike"].notna().all()


def test_reasons_accumulate() -> None:
    returns, proxy = simulated_panel(n=300)
    returns, proxy = returns.copy(), proxy.copy()
    splitter = make_splitter()
    target = int(list(splitter.split(returns.size))[2].test[0])
    returns[target] = np.nan
    proxy[target] = np.nan
    frame = backtest(OracleNormal(SIGMA), returns, proxy, splitter)
    row = frame.loc[frame["target_index"] == target].iloc[0]
    assert row["missing_reason"] == "proxy_nan|target_nan"


# --------------------------------------------------------------------------
# refit cadence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("refit_every", [1, 2, 5, 21, 1000])
def test_refit_cadence_is_honoured(refit_every: int) -> None:
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=refit_every)
    spy = SpyModel()

    frame = backtest(spy, returns, proxy, splitter)

    origins = list(splitter.split(returns.size))
    expected_fits = sum(1 for o in origins if o.refit)
    assert expected_fits == -(-len(origins) // refit_every)  # ceil, from the splitter
    assert spy.fit_calls == expected_fits
    assert spy.predict_calls == len(origins)
    assert frame["refit"].sum() == expected_fits


def test_each_fit_sees_exactly_its_splitter_training_window() -> None:
    """The splitter-monopoly check: what the model was handed, verbatim."""
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=7)
    spy = SpyModel()
    backtest(spy, returns, proxy, splitter)

    refit_origins = [o for o in splitter.split(returns.size) if o.refit]
    assert len(spy.fit_windows) == len(refit_origins)
    for window, origin in zip(spy.fit_windows, refit_origins, strict=True):
        assert np.array_equal(window, returns[origin.train])
        assert origin.train.max() == origin.origin  # nothing past the cutoff


def test_between_refits_the_fit_is_reused_and_provenance_says_so() -> None:
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=10)
    frame = backtest(SpyModel(), returns, proxy, splitter)

    # Parameters always come from an origin at or before the forecast origin.
    assert (frame["fit_origin"] <= frame["origin_index"]).all()
    assert (
        frame.loc[frame["refit"], "fit_origin"] == frame.loc[frame["refit"], "origin_index"]
    ).all()
    # Without update support, conditioning is held at the last refit.
    assert (frame["conditioned_through"] == frame["fit_origin"]).all()
    # A reused fit gives a repeated forecast — the cost this records.
    assert frame["forecast_var"].nunique() == int(frame["refit"].sum())


def test_a_model_that_supports_update_re_conditions_without_re_estimating() -> None:
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=10)
    model = UpdatingModel()
    frame = backtest(model, returns, proxy, splitter)

    origins = list(splitter.split(returns.size))
    n_refits = sum(1 for o in origins if o.refit)
    assert model.fit_calls == n_refits  # cadence still respected
    assert model.update_calls == len(origins) - n_refits
    assert (frame["conditioned_through"] == frame["origin_index"]).all()
    assert frame["forecast_var"].nunique() == len(origins)


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------


def test_future_data_cannot_touch_earlier_forecasts() -> None:
    """THE leakage canary (`.claude/skills/leakage-check`, final paragraph).

    Corrupt everything strictly after a cutoff T; every forecast for a target
    at or before T must be bit-identical. Only ``config_hash`` may differ —
    it hashes the data by content, which is exactly why a poisoned series can
    never be served from the clean one's cache.
    """
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=5)
    cutoff = int(list(splitter.split(returns.size))[150].test[-1])

    poisoned_returns, poisoned_proxy = returns.copy(), proxy.copy()
    poisoned_returns[cutoff + 1 :] = 1e9
    poisoned_proxy[cutoff + 1 :] = 1e9

    clean = backtest(SpyModel(), returns, proxy, splitter)
    poisoned = backtest(SpyModel(), poisoned_returns, poisoned_proxy, splitter)

    scored = ["crps", "log_score", "qlike", "forecast_mean", "forecast_var", "realized_return"]
    scored += [f"{p}_{lv:.10g}".replace(".", "p") for lv in DEFAULT_LEVELS for p in ("var", "hit")]
    before = clean.loc[clean["target_index"] <= cutoff, scored].reset_index(drop=True)
    after = poisoned.loc[poisoned["target_index"] <= cutoff, scored].reset_index(drop=True)

    assert len(before) > 100
    pd.testing.assert_frame_equal(before, after)
    assert clean.attrs["config_hash"] != poisoned.attrs["config_hash"]

    # A canary that cannot die is not a canary: prove the poison was potent by
    # checking it did reach every forecast whose target is after the cutoff.
    later_clean = clean.loc[clean["target_index"] > cutoff, "crps"].to_numpy()
    later_poisoned = poisoned.loc[poisoned["target_index"] > cutoff, "crps"].to_numpy()
    assert len(later_clean) > 100
    assert (later_clean != later_poisoned).all()


def test_the_proxy_never_reaches_the_model() -> None:
    """QLIKE's target is a realized quantity; if it could reach ``fit`` the
    variance forecasts would be scoring themselves."""
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter()
    baseline = backtest(SpyModel(), returns, proxy, splitter)
    nonsense = backtest(SpyModel(), returns, np.full_like(proxy, 4.2), splitter)
    pd.testing.assert_frame_equal(
        baseline[["forecast_mean", "forecast_var"]], nonsense[["forecast_mean", "forecast_var"]]
    )


def test_fit_series_is_sliced_with_the_same_splitter_indices() -> None:
    """HAR-RV fits on realized variances but is scored on returns; both must
    be read through the same origins."""
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter(refit_every=3)
    spy = SpyModel()
    frame = backtest(spy, returns, proxy, splitter, fit_series=proxy)

    refit_origins = [o for o in splitter.split(returns.size) if o.refit]
    for window, origin in zip(spy.fit_windows, refit_origins, strict=True):
        assert np.array_equal(window, proxy[origin.train])
    # Scoring still uses the return series.
    assert np.array_equal(
        frame["realized_return"].to_numpy(), returns[frame["target_index"].to_numpy()]
    )


def test_changing_the_data_changes_the_config_hash() -> None:
    """Leakage-check item 9: a cached artifact must never serve different data."""
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter()
    base = backtest(OracleNormal(SIGMA), returns, proxy, splitter)

    nudged = returns.copy()
    nudged[0] += 1e-12
    assert (
        backtest(OracleNormal(SIGMA), nudged, proxy, splitter).attrs["config_hash"]
        != (base.attrs["config_hash"])
    )
    other_proxy = backtest(OracleNormal(SIGMA), returns, proxy * 1.01, splitter)
    assert other_proxy.attrs["config_hash"] != base.attrs["config_hash"]


# --------------------------------------------------------------------------
# execution seam
# --------------------------------------------------------------------------


def test_work_is_routed_through_the_executor_one_item_per_refit_block() -> None:
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=21)
    executor = CountingExecutor()
    backtest(SpyModel(), returns, proxy, splitter, executor=executor)

    origins = list(splitter.split(returns.size))
    assert executor.map_calls == 1
    assert executor.n_items == sum(1 for o in origins if o.refit)


@pytest.mark.parametrize("executor_factory", [SerialExecutor, ShuffledExecutor, OutOfOrderExecutor])
def test_results_do_not_depend_on_execution_order(executor_factory: Callable[[], Any]) -> None:
    """D-011's backend-invariance claim in miniature."""
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=7)
    reference = backtest(SpyModel(), returns, proxy, splitter, executor=SerialExecutor())
    other = backtest(SpyModel(), returns, proxy, splitter, executor=executor_factory())
    pd.testing.assert_frame_equal(reference, other)


# --------------------------------------------------------------------------
# store integration and determinism
# --------------------------------------------------------------------------


def test_a_cached_config_short_circuits_the_run(tmp_path: Path) -> None:
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=5)
    store = ResultsStore(tmp_path / "results")

    first_spy = SpyModel()
    first = backtest(first_spy, returns, proxy, splitter, store=store)
    assert first_spy.fit_calls > 0
    assert first.attrs["cached"] is False

    second_spy = SpyModel()
    second = backtest(second_spy, returns, proxy, splitter, store=store)
    assert second_spy.fit_calls == 0  # no fitting
    assert second_spy.predict_calls == 0  # no forecasting
    assert second.attrs["cached"] is True
    pd.testing.assert_frame_equal(first, second)


def test_rerunning_does_not_duplicate_stored_rows(tmp_path: Path) -> None:
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter()
    store = ResultsStore(tmp_path / "results")
    for _ in range(3):
        backtest(SpyModel(), returns, proxy, splitter, store=store)
    stored = store.read_all()
    assert len(stored) == splitter.n_splits(returns.size)
    assert not stored.duplicated(subset=["config_hash", "origin_index", "horizon"]).any()


def test_overwrite_recomputes_instead_of_reading_the_cache(tmp_path: Path) -> None:
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter()
    store = ResultsStore(tmp_path / "results")
    backtest(SpyModel(), returns, proxy, splitter, store=store)
    spy = SpyModel()
    frame = backtest(spy, returns, proxy, splitter, store=store, overwrite=True)
    assert spy.fit_calls > 0
    assert frame.attrs["cached"] is False


def test_different_configs_coexist_in_one_store(tmp_path: Path) -> None:
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter()
    store = ResultsStore(tmp_path / "results")
    backtest(OracleNormal(SIGMA, label="a"), returns, proxy, splitter, store=store)
    backtest(OracleNormal(2 * SIGMA, label="b"), returns, proxy, splitter, store=store)
    assert len(store.config_hashes()) == 2
    assert set(store.read_all()["model"]) == {"a", "b"}


def test_the_stored_config_reproduces_the_hash(tmp_path: Path) -> None:
    from volbench.results import config_hash

    returns, proxy = simulated_panel(n=400)
    store = ResultsStore(tmp_path / "results")
    frame = backtest(OracleNormal(SIGMA), returns, proxy, make_splitter(), store=store)
    digest = frame.attrs["config_hash"]
    assert config_hash(frame.attrs["config"]) == digest
    assert store.read_config(digest)["splitter"]["window"] == 250


def test_two_identical_runs_produce_identical_parquet(tmp_path: Path) -> None:
    """THE determinism canary (CLAUDE.md rule 3).

    Not just equal frames — equal bytes on disk, which is what `make
    reproduce` and the cross-backend identity claim actually promise.
    """
    returns, proxy = simulated_panel(n=600)
    splitter = make_splitter(refit_every=5)

    frames = []
    stores = []
    for run in ("a", "b"):
        store = ResultsStore(tmp_path / run)
        frames.append(backtest(SpyModel(), returns, proxy, splitter, store=store))
        stores.append(store)

    pd.testing.assert_frame_equal(frames[0], frames[1])
    assert frames[0].attrs["config_hash"] == frames[1].attrs["config_hash"]

    digest = frames[0].attrs["config_hash"]
    assert (
        stores[0].fragment_path(digest).read_bytes() == stores[1].fragment_path(digest).read_bytes()
    )
    assert stores[0].config_path(digest).read_bytes() == stores[1].config_path(digest).read_bytes()


# --------------------------------------------------------------------------
# forecast moments
# --------------------------------------------------------------------------


def test_moments_of_a_normal_are_exact() -> None:
    assert forecast_moments(Distribution.from_normal(0.3, 2.0)) == (0.3, 4.0)


def test_moments_of_an_ensemble_are_the_plug_in_moments() -> None:
    samples = np.array([-1.0, 0.0, 0.5, 2.0])
    mean, variance = forecast_moments(Distribution.from_samples(samples))
    assert mean == pytest.approx(float(np.mean(samples)))
    assert variance == pytest.approx(float(np.var(samples)))


def test_moments_of_a_quantile_grid_are_close_and_conservative() -> None:
    """Flat extrapolation past the outermost tau truncates tail mass, so the
    variance is slightly understated — documented, and bounded here."""
    normal = Normal(0.0, 1.0)
    taus = np.linspace(0.001, 0.999, 999)
    grid = Distribution.from_quantiles(taus, np.array([normal.quantile(float(t)) for t in taus]))
    mean, variance = forecast_moments(grid)
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert variance == pytest.approx(1.0, rel=0.01)
    assert variance < 1.0


def test_moments_fall_back_to_quadrature_for_unknown_families() -> None:
    @dataclass(frozen=True, eq=False)
    class Uniform01(Distribution):
        def quantile(self, tau: float) -> float:
            return tau

    mean, variance = forecast_moments(Uniform01())
    assert mean == pytest.approx(0.5, abs=1e-6)
    assert variance == pytest.approx(1.0 / 12.0, abs=1e-6)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_misaligned_proxy_is_rejected() -> None:
    returns, proxy = simulated_panel(n=400)
    with pytest.raises(ValueError, match="expected 400"):
        backtest(OracleNormal(SIGMA), returns, proxy[:-1], make_splitter())


# --------------------------------------------------------------------------
# one calendar (M1 report §4.6)
# --------------------------------------------------------------------------
#
# ``run_backtest`` aligns by position. Two same-length inputs on different
# calendars would score every forecast against the wrong day's realization —
# leakage no other test can see, because every score would still be a
# perfectly plausible number. Pandas inputs carry the calendar, so the guard
# can be structural: identical indexes, or no run.


def _dated(values: NDArray[np.float64], start: str = "2020-01-01") -> pd.Series:
    index = pd.date_range(start, periods=values.size, freq="B", tz="UTC")
    return pd.Series(values, index=index)


def test_inputs_shifted_by_one_day_are_rejected_naming_the_first_mismatch() -> None:
    """The headline case: same length, same values, calendars one day apart."""
    returns, proxy = simulated_panel(n=400)
    dated_returns = _dated(returns)
    shifted_proxy = pd.Series(proxy, index=dated_returns.index.shift(1, freq="B"))

    with pytest.raises(ValueError, match="first mismatch at position 0") as info:
        backtest(OracleNormal(SIGMA), dated_returns, shifted_proxy, make_splitter())

    message = str(info.value)
    # Names both inputs, both offending timestamps, and why it matters.
    assert "proxy is not on the same calendar as series" in message
    assert str(dated_returns.index[0]) in message
    assert str(shifted_proxy.index[0]) in message
    assert "wrong day's realization" in message


def test_a_shift_deep_in_the_series_names_that_position() -> None:
    returns, proxy = simulated_panel(n=400)
    dated_returns = _dated(returns)
    calendar = dated_returns.index.to_list()
    calendar[123] = calendar[123] + pd.Timedelta(hours=1)
    off_by_an_hour = pd.Series(proxy, index=pd.DatetimeIndex(calendar))

    with pytest.raises(ValueError, match="first mismatch at position 123"):
        backtest(OracleNormal(SIGMA), dated_returns, off_by_an_hour, make_splitter())


def test_fit_series_must_share_the_calendar_too() -> None:
    returns, proxy = simulated_panel(n=400)
    dated_returns, dated_proxy = _dated(returns), _dated(proxy)
    shifted_fit = pd.Series(proxy, index=dated_returns.index.shift(1, freq="B"))

    with pytest.raises(ValueError, match="fit_series is not on the same calendar as series"):
        backtest(
            OracleNormal(SIGMA), dated_returns, dated_proxy, make_splitter(), fit_series=shifted_fit
        )


def test_an_input_that_ran_out_early_is_named_as_such() -> None:
    returns, proxy = simulated_panel(n=400)
    dated_returns = _dated(returns)
    truncated_proxy = _dated(proxy).iloc[:-1]

    with pytest.raises(ValueError, match=r"position 399, where proxy has run out"):
        backtest(OracleNormal(SIGMA), dated_returns, truncated_proxy, make_splitter())


def test_a_calendar_cannot_be_compared_with_positions() -> None:
    """Dropping the index on one side (a stray ``.to_numpy()`` then
    ``pd.Series``) leaves a RangeIndex, which is not a calendar."""
    returns, proxy = simulated_panel(n=400)
    with pytest.raises(ValueError, match="first mismatch at position 0"):
        backtest(OracleNormal(SIGMA), _dated(returns), pd.Series(proxy), make_splitter())


def test_mixing_indexed_and_bare_inputs_is_refused() -> None:
    returns, proxy = simulated_panel(n=400)
    with pytest.raises(ValueError, match="proxy is a bare array"):
        backtest(OracleNormal(SIGMA), _dated(returns), proxy, make_splitter())
    with pytest.raises(ValueError, match="fit_series is a bare array"):
        backtest(
            OracleNormal(SIGMA), _dated(returns), _dated(proxy), make_splitter(), fit_series=proxy
        )


def test_inputs_on_one_calendar_score_exactly_as_bare_arrays_do() -> None:
    """The guard validates the calendar; it does not change a single number,
    and it does not enter the config hash — same values, same experiment."""
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter(refit_every=5)
    bare = backtest(SpyModel(), returns, proxy, splitter)
    dated = backtest(SpyModel(), _dated(returns), _dated(proxy), splitter)
    pd.testing.assert_frame_equal(bare, dated)
    assert bare.attrs["config_hash"] == dated.attrs["config_hash"]


def test_bare_arrays_remain_an_explicit_positional_opt_in() -> None:
    """No calendar to check means nothing to check; only lengths are compared.
    This is the README quickstart's path and must keep working."""
    returns, proxy = simulated_panel(n=400)
    frame = backtest(OracleNormal(SIGMA), returns, proxy, make_splitter())
    assert len(frame) == make_splitter().n_splits(returns.size)


def test_bad_levels_are_rejected() -> None:
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter()
    for levels, match in (
        ((), "must not be empty"),
        ((0.01, 0.01), "must be distinct"),
        ((0.0, 0.05), "strictly inside"),
        ((0.05, 1.0), "strictly inside"),
    ):
        with pytest.raises(ValueError, match=match):
            backtest(OracleNormal(SIGMA), returns, proxy, splitter, levels=levels)


def test_levels_are_configurable_and_reach_the_columns_and_the_hash() -> None:
    returns, proxy = simulated_panel(n=400)
    splitter = make_splitter()
    default = backtest(OracleNormal(SIGMA), returns, proxy, splitter)
    custom = backtest(OracleNormal(SIGMA), returns, proxy, splitter, levels=(0.1,))
    assert "hit_0p1" in custom.columns
    assert "hit_0p01" not in custom.columns
    assert custom.attrs["config_hash"] != default.attrs["config_hash"]


def test_a_series_too_short_for_the_splitter_is_rejected_by_the_splitter() -> None:
    returns, proxy = simulated_panel(n=250)
    with pytest.raises(ValueError, match="too short"):
        backtest(OracleNormal(SIGMA), returns, proxy, make_splitter())
