"""The shared TSFM adapter contract, exercised without weights (runs in CI).

What is pinned here holds for every foundation-model adapter at once, because
they all go through ``tsfm_common``: the context ends at the origin, the
variance forecast is the mean of the model's RV quantile grid **under the
lognormal tail closure** (``TestTailClosure``; the unclosed reading is still
pinned equal to the one the evaluator would take from a ``QuantileGrid``), the
scored object is a ``Normal(0, sqrt(vhat))`` over the return, ``update`` is
exact context extension, ``input_scale`` round-trips, and a backtest cannot
see the future.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from tsfm_fakes import DEFAULT_TAUS, FAKE_SHA, FakeBackend, ScriptedBackend, realized_variance

from volbench import evaluate as evaluate_module
from volbench.dist import Normal
from volbench.evaluate import SupportsUpdate, run_backtest
from volbench.models import Chronos, Moirai, TimeGPT, TimesFM
from volbench.models.base import FittedModel, ForecastModel
from volbench.models.tsfm_common import (
    CLOSURES,
    MIN_CONTEXT,
    VARIANCE_FROM,
    FittedTSFM,
    TSFMBackend,
    ZeroShotRVModel,
    checkpoint_slug,
    grid_mean_under_closures,
    quantile_grid_mean,
    rearrange_quantiles,
    resolve_hf_revision,
    tail_closed_grid_mean,
    validated_rv,
)
from volbench.splitter import RollingOriginSplitter

ADAPTERS = [Chronos, TimesFM, Moirai, TimeGPT]


def _model(cls: type[ZeroShotRVModel], **kwargs: object) -> ZeroShotRVModel:
    return cls(backend=FakeBackend(), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------


class TestQuantileGridMean:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize(
        "taus",
        [DEFAULT_TAUS, (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99), (0.25, 0.75)],
    )
    def test_matches_the_evaluators_quantile_grid_mean(
        self, seed: int, taus: tuple[float, ...]
    ) -> None:
        rng = np.random.default_rng(seed)
        t = np.asarray(taus)
        v = np.sort(rng.lognormal(np.log(1e-4), 0.6, size=t.size))
        ours = quantile_grid_mean(t, v)
        theirs, _ = evaluate_module._moments_from_quantile_grid(t, v)
        assert ours == theirs  # same formula, bit for bit

    def test_degenerate_grid_is_its_value(self) -> None:
        assert quantile_grid_mean(np.asarray(DEFAULT_TAUS), np.full(9, 3.0)) == pytest.approx(3.0)

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            quantile_grid_mean(np.asarray([0.5]), np.asarray([1.0]))
        with pytest.raises(ValueError):
            quantile_grid_mean(np.asarray([0.1, 0.9]), np.asarray([1.0, 2.0, 3.0]))


class TestTailClosure:
    """The fix of docs/P3_TSFM_VARIANCE_AUDIT.md: the mean was the right
    functional, the flat tails were the bug."""

    def test_flat_is_exactly_the_unclosed_grid_mean(self) -> None:
        t = np.asarray(DEFAULT_TAUS)
        v = np.sort(np.random.default_rng(0).lognormal(np.log(1e-4), 0.6, size=t.size))
        assert tail_closed_grid_mean(t, v, "flat") == quantile_grid_mean(t, v)

    @pytest.mark.parametrize("closure", ["lognormal", "loglinear"])
    def test_closing_the_tails_of_a_right_skewed_grid_raises_the_mean(
        self, closure: str
    ) -> None:
        """The direction is the whole finding: 20 % of the mass sits in two
        atoms, and on a right-skewed law reading them as atoms understates."""
        t = np.asarray(DEFAULT_TAUS)
        v = np.exp(np.log(1e-4) + 0.6 * stats.norm.ppf(t))
        assert tail_closed_grid_mean(t, v, closure) > quantile_grid_mean(t, v)

    def test_on_an_exactly_lognormal_grid_the_closure_recovers_its_mean(self) -> None:
        """The closure is exact where its assumption is exactly true, up to the
        interior trapezoid — which is the only approximation left in it."""
        mu, sigma = math.log(1e-4), 0.6
        t = np.asarray(DEFAULT_TAUS)
        v = np.exp(mu + sigma * stats.norm.ppf(t))
        assert tail_closed_grid_mean(t, v, "lognormal") == pytest.approx(
            math.exp(mu + 0.5 * sigma**2), rel=0.02
        )

    def test_a_grid_holding_a_zero_cannot_be_closed(self) -> None:
        t = np.asarray(DEFAULT_TAUS)
        v = np.asarray([0.0, 0.0, 0.4, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        assert math.isnan(tail_closed_grid_mean(t, v, "lognormal"))
        assert math.isfinite(tail_closed_grid_mean(t, v, "flat"))

    def test_a_degenerate_grid_cannot_be_closed_either(self) -> None:
        t = np.asarray(DEFAULT_TAUS)
        assert math.isnan(tail_closed_grid_mean(t, np.full(t.size, 3.0), "lognormal"))

    def test_the_sensitivity_reports_every_closure_and_names_them(self) -> None:
        t = np.asarray(DEFAULT_TAUS)
        v = np.exp(np.log(1e-4) + 0.6 * stats.norm.ppf(t))
        out = grid_mean_under_closures(t, v)
        assert tuple(out) == CLOSURES == ("flat", "lognormal", "loglinear")
        assert out["flat"] < out["loglinear"]
        assert out["flat"] < out["lognormal"]

    def test_an_unknown_closure_is_refused(self) -> None:
        t = np.asarray(DEFAULT_TAUS)
        with pytest.raises(ValueError, match="unknown closure"):
            tail_closed_grid_mean(t, np.arange(1.0, 10.0), "student")

    def test_rejects_bad_shapes_like_the_unclosed_mean(self) -> None:
        with pytest.raises(ValueError):
            tail_closed_grid_mean(np.asarray([0.5]), np.asarray([1.0]))

    def test_the_closure_is_named_in_spec_and_therefore_in_the_config_hash(self) -> None:
        """The label must name the estimator: leaving it at
        ``mean_of_rv_quantile_grid`` while changing how that mean is computed
        would make every sidecar a false statement about its own numbers, and
        the store would serve pre-fix fragments for post-fix configs."""
        spec = _model(Chronos).spec()
        assert spec["variance_from"] == VARIANCE_FROM == (
            "lognormal_tail_closed_mean_of_rv_quantile_grid"
        )
        assert "tail_closed" in spec["variance_from"]


class TestRearrangeQuantiles:
    def test_sorts_and_counts_crossings(self) -> None:
        out, n = rearrange_quantiles(np.asarray([1.0, 3.0, 2.0, 5.0, 4.0]))
        assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert n == 2

    def test_monotone_input_is_copied_not_aliased(self) -> None:
        v = np.asarray([1.0, 2.0, 2.0, 3.0])
        out, n = rearrange_quantiles(v)
        assert n == 0 and out.tolist() == v.tolist() and out is not v


class TestSlugAndValidation:
    def test_checkpoint_slug(self) -> None:
        assert checkpoint_slug("chronos", "amazon/chronos-bolt-small") == "chronos_bolt_small"
        assert checkpoint_slug("chronos", "amazon/chronos-2") == "chronos_2"
        assert checkpoint_slug("timesfm", "google/timesfm-2.5-200m-pytorch") == (
            "timesfm_2_5_200m_pytorch"
        )
        assert checkpoint_slug("moirai", "Salesforce/moirai-2.0-R-small") == "moirai_2_0_r_small"
        assert checkpoint_slug("timegpt", "timegpt-1") == "timegpt_1"

    def test_validated_rv_accepts_zeros_and_rejects_the_rest(self) -> None:
        ok = np.zeros(MIN_CONTEXT)
        assert validated_rv(ok).size == MIN_CONTEXT
        with pytest.raises(ValueError, match="at least"):
            validated_rv(np.ones(MIN_CONTEXT - 1))
        with pytest.raises(ValueError, match="1-D"):
            validated_rv(np.ones((MIN_CONTEXT, 2)))
        bad = np.ones(MIN_CONTEXT)
        bad[3] = -1e-9
        with pytest.raises(ValueError, match="non-negative"):
            validated_rv(bad)
        bad[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            validated_rv(bad)


class TestResolveHfRevision:
    def test_commit_hash_passes_through_without_touching_the_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("huggingface_hub")
        sha = "0123456789abcdef0123456789abcdef01234567"
        monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent/path")
        assert resolve_hf_revision("org/model", sha) == sha

    def test_reads_the_ref_the_cache_holds(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hub = pytest.importorskip("huggingface_hub")
        from huggingface_hub.file_download import repo_folder_name

        monkeypatch.setattr(hub.constants, "HF_HUB_CACHE", str(tmp_path))
        refs = tmp_path / repo_folder_name(repo_id="org/model", repo_type="model") / "refs"  # type: ignore[operator]
        refs.mkdir(parents=True)
        (refs / "main").write_text(FAKE_SHA)
        (refs / "v1").write_text("1" * 40)
        assert resolve_hf_revision("org/model") == FAKE_SHA
        assert resolve_hf_revision("org/model", "v1") == "1" * 40
        with pytest.raises(RuntimeError, match="not in the local Hugging Face cache"):
            resolve_hf_revision("org/model", "v2")
        (refs / "main").write_text("not a sha")
        with pytest.raises(RuntimeError, match="unexpected ref content"):
            resolve_hf_revision("org/model")


# --------------------------------------------------------------------------
# the adapter contract, per concrete class
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ADAPTERS, ids=lambda c: c.__name__)
class TestProtocolConformance:
    def test_satisfies_the_model_protocols(self, cls: type[ZeroShotRVModel]) -> None:
        model = _model(cls)
        assert isinstance(model, ForecastModel)
        assert isinstance(model.backend, TSFMBackend)
        fitted = model.fit(realized_variance())
        assert isinstance(fitted, FittedModel)
        assert isinstance(fitted, SupportsUpdate)
        assert isinstance(fitted, FittedTSFM)
        assert fitted.name == model.name

    def test_spec_is_json_stable_and_hyperparameter_sensitive(
        self, cls: type[ZeroShotRVModel]
    ) -> None:
        a, b = _model(cls), _model(cls)
        json.dumps(a.spec(), sort_keys=True)
        assert a.spec() == b.spec()
        assert a.spec()["model"] == a.name
        assert a.spec()["zero_shot"] is True
        assert a.spec()["revision"] == FAKE_SHA
        assert "backend" not in {k for k, v in a.spec().items() if v is a.backend}
        assert _model(cls, context_length=256).spec() != a.spec()
        assert _model(cls, input_scale=1.0).spec() != a.spec()

    def test_constructor_validation(self, cls: type[ZeroShotRVModel]) -> None:
        with pytest.raises(ValueError, match="context_length"):
            _model(cls, context_length=MIN_CONTEXT - 1)
        with pytest.raises(ValueError, match="input_scale"):
            _model(cls, input_scale=0.0)
        with pytest.raises(ValueError, match="input_scale"):
            _model(cls, input_scale=math.inf)


class TestContextConstruction:
    """Leakage-check item 7: the context ends at the origin and never touches t+1."""

    def test_fit_records_exactly_the_trailing_window(self) -> None:
        rv = realized_variance(600)
        fitted = Chronos(backend=FakeBackend(), context_length=100).fit(rv)
        assert np.array_equal(fitted.context, rv[-100:])
        assert fitted.context[-1] == rv[-1]
        assert fitted.context.size == 100

    def test_default_context_is_the_whole_window_capped_at_the_backend_max(self) -> None:
        rv = realized_variance(600)
        assert Chronos(backend=FakeBackend(max_context=2048)).fit(rv).context.size == 600
        assert Chronos(backend=FakeBackend(max_context=512)).fit(rv).context.size == 512
        assert np.array_equal(
            Chronos(backend=FakeBackend(max_context=512)).fit(rv).context, rv[-512:]
        )

    def test_a_shorter_window_than_context_length_is_used_whole(self) -> None:
        rv = realized_variance(80)
        fitted = Chronos(backend=FakeBackend(), context_length=1000).fit(rv)
        assert np.array_equal(fitted.context, rv)

    def test_the_backend_sees_only_the_scaled_context(self) -> None:
        rv = realized_variance(300)
        backend = FakeBackend()
        fitted = Chronos(backend=backend, context_length=64, input_scale=1e4).fit(rv)
        fitted.predict(1)
        (seen, h), *_ = backend.calls
        assert h == 1
        assert np.array_equal(seen, rv[-64:] * 1e4)

    def test_fit_and_update_copy_their_input(self) -> None:
        rv = realized_variance(300)
        fitted = Chronos(backend=FakeBackend()).fit(rv)
        rv[-1] = 123.0
        assert fitted.context[-1] != 123.0


class TestPredict:
    def test_returns_normal_over_the_return_with_the_tail_closed_grid_mean(self) -> None:
        rv = realized_variance(300)
        backend = FakeBackend()
        fitted = Chronos(backend=backend, input_scale=1.0).fit(rv)
        dist = fitted.predict(1)
        assert isinstance(dist, Normal)
        assert dist.mu == 0.0
        expected_grid = backend.forecast(fitted.context, 1).values[0]
        taus = np.asarray(DEFAULT_TAUS)
        vhat = tail_closed_grid_mean(taus, expected_grid)
        assert dist.variance() == pytest.approx(vhat, rel=1e-12)
        assert dist.sigma == math.sqrt(vhat)
        # daily units, never annualized
        assert 1e-6 < vhat < 1e-2
        # and it is strictly the flat reading plus the closed tails
        assert vhat > quantile_grid_mean(taus, expected_grid)

    def test_step_h_uses_row_h_minus_one(self) -> None:
        rv = realized_variance(300)
        backend = FakeBackend()
        fitted = Chronos(backend=backend, input_scale=1.0).fit(rv)
        v1, v3 = fitted.predict(1).variance(), fitted.predict(3).variance()
        rows = backend.forecast(fitted.context, 3).values
        taus = np.asarray(DEFAULT_TAUS)
        assert v3 == pytest.approx(tail_closed_grid_mean(taus, rows[2]), rel=1e-12)
        assert v3 == pytest.approx(v1 * 1.03 / 1.01, rel=1e-12)
        with pytest.raises(ValueError, match="h must be >= 1"):
            fitted.predict(0)

    def test_input_scale_round_trips(self) -> None:
        rv = realized_variance(300)
        v1 = Chronos(backend=FakeBackend(), input_scale=1.0).fit(rv).predict(1).variance()
        v4 = Chronos(backend=FakeBackend(), input_scale=1e4).fit(rv).predict(1).variance()
        v7 = Chronos(backend=FakeBackend(), input_scale=1e7).fit(rv).predict(1).variance()
        assert v4 == pytest.approx(v1, rel=1e-12)
        assert v7 == pytest.approx(v1, rel=1e-12)

    def test_native_mean_is_recorded_in_units_but_never_scored(self) -> None:
        rv = realized_variance(300)
        backend = FakeBackend(native_mean=True)
        fitted = Chronos(backend=backend, input_scale=1e4).fit(rv)
        dist = fitted.predict(2)
        meta = fitted.spec()["rv_forecasts"]["2"]
        expected_native = backend.level_of(fitted.context) * math.exp(0.125) * 1.02
        assert meta["native_mean"] == pytest.approx(expected_native, rel=1e-12)
        assert dist.variance() == pytest.approx(meta["mean"], rel=1e-12)
        assert meta["mean"] != pytest.approx(meta["native_mean"], rel=1e-3)

    def test_crossing_is_rearranged_and_negatives_clipped_and_counted(self) -> None:
        rows = np.asarray([[-2.0, -1.0, 0.5, 0.4, 1.0, 2.0, 1.5, 3.0, 4.0]])
        fitted = Chronos(backend=ScriptedBackend(rows), input_scale=1.0).fit(realized_variance(100))
        dist = fitted.predict(1)
        meta = fitted.spec()["rv_forecasts"]["1"]
        assert meta["crossings_rearranged"] == 2
        assert meta["clipped_at_zero"] == 2
        grid = np.asarray(meta["values"])
        assert grid.tolist() == [0.0, 0.0, 0.4, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        # A clipped zero is exactly the state no lognormal can describe, so
        # this origin keeps the flat reading — and says so.
        assert meta["tail_closure"] == "flat"
        assert dist.variance() == pytest.approx(quantile_grid_mean(np.asarray(DEFAULT_TAUS), grid))
        assert dist.variance() == meta["flat_tail_mean"]

    def test_non_positive_mean_is_a_predict_error_not_a_forecast(self) -> None:
        rows = np.asarray([[-1.0] * 9])
        fitted = Chronos(backend=ScriptedBackend(rows), input_scale=1.0).fit(realized_variance(100))
        with pytest.raises(ValueError, match="not positive"):
            fitted.predict(1)

    def test_wrong_shape_from_the_backend_is_loud(self) -> None:
        rows = np.ones((1, 9))
        fitted = Chronos(backend=ScriptedBackend(rows), input_scale=1.0).fit(realized_variance(100))
        with pytest.raises(RuntimeError, match="shape"):
            fitted.predict(2)  # scripted with one row, asked for two

    def test_forecasts_are_memoised_per_horizon(self) -> None:
        backend = FakeBackend()
        fitted = Chronos(backend=backend).fit(realized_variance(100))
        fitted.predict(1)
        fitted.predict(1)
        fitted.predict(2)
        assert [h for _, h in backend.calls] == [1, 2]

    def test_fitted_spec_is_json_and_cheap(self) -> None:
        fitted = Chronos(backend=FakeBackend()).fit(realized_variance(100))
        before = fitted.spec()
        assert before["n_context"] == 100 and before["rv_forecasts"] == {}
        json.dumps(before, sort_keys=True)
        fitted.predict(1)
        after = fitted.spec()
        assert set(after["rv_forecasts"]) == {"1"}
        assert after["rv_forecasts"]["1"]["taus"] == list(DEFAULT_TAUS)
        json.dumps(after, sort_keys=True)


class TestUpdate:
    def test_update_on_the_fit_window_is_the_fit(self) -> None:
        rv = realized_variance(600)
        fitted = Chronos(backend=FakeBackend(), context_length=200).fit(rv[:500])
        again = fitted.update(rv[:500])
        assert np.array_equal(again.context, fitted.context)
        assert again.predict(1) == fitted.predict(1)  # Normal: value equality
        assert again.backend is fitted.backend

    def test_update_extends_the_context_exactly(self) -> None:
        rv = realized_variance(600)
        fitted = Chronos(backend=FakeBackend(), context_length=200).fit(rv[:500])
        moved = fitted.update(rv[1:501])
        assert np.array_equal(moved.context, rv[301:501])
        assert moved.context[-1] == rv[500]
        assert moved.predict(1) != fitted.predict(1)

    def test_update_is_what_fit_would_have_done(self) -> None:
        rv = realized_variance(600)
        model = Chronos(backend=FakeBackend(), context_length=200)
        via_update = model.fit(rv[:500]).update(rv[7:507])
        via_fit = model.fit(rv[7:507])
        assert np.array_equal(via_update.context, via_fit.context)
        assert via_update.predict(1) == via_fit.predict(1)

    def test_update_rejects_what_fit_rejects(self) -> None:
        fitted = Chronos(backend=FakeBackend()).fit(realized_variance(300))
        bad = realized_variance(300)
        bad[10] = -1.0
        with pytest.raises(ValueError):
            fitted.update(bad)


# --------------------------------------------------------------------------
# through the evaluator
# --------------------------------------------------------------------------


def _panel(n: int = 260, seed: int = 3) -> tuple[pd.Series, pd.Series]:
    """Returns and a matching realized-variance series on one calendar."""
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(np.log(1e-4), 0.4, size=n))
    r = np.sqrt(rv) * rng.standard_normal(n)
    index = pd.RangeIndex(n)
    return pd.Series(r, index=index), pd.Series(rv, index=index)


def _run(returns: pd.Series, rv: pd.Series, *, refit_every: int = 1) -> pd.DataFrame:
    splitter = RollingOriginSplitter(window=100, horizon=1, step=1, refit_every=refit_every)
    return run_backtest(
        lambda: Chronos(backend=FakeBackend(), context_length=64),
        returns,
        rv,
        splitter,
        0,
        asset="SIM",
        proxy_name="rv",
        fit_series=rv,
    )


class TestThroughTheEvaluator:
    def test_every_origin_is_conditioned_through_itself(self) -> None:
        out = _run(*_panel())
        assert (out["missing_reason"] == "").all()
        assert (out["conditioned_through"] == out["origin_index"]).all()
        assert (out["target_index"] == out["origin_index"] + 1).all()

    def test_refit_every_is_irrelevant_to_a_zero_shot_model(self) -> None:
        """Refit or update, the context is the trailing window either way."""
        returns, rv = _panel()
        every = _run(returns, rv, refit_every=1)
        monthly = _run(returns, rv, refit_every=21)
        scores = [c for c in every.columns if c not in {"config_hash", "refit", "fit_origin"}]
        pd.testing.assert_frame_equal(
            every[scores].reset_index(drop=True),
            monthly[scores].reset_index(drop=True),
            check_exact=True,
        )
        assert monthly["refit"].sum() < every["refit"].sum()

    def test_leakage_canary_future_corruption_cannot_change_past_forecasts(self) -> None:
        returns, rv = _panel()
        cutoff = 100 + 60
        clean = _run(returns, rv)
        rng = np.random.default_rng(999)
        dirty_rv = rv.copy()
        dirty_rv.iloc[cutoff + 1 :] = np.exp(rng.normal(-5.0, 1.0, size=rv.size - cutoff - 1))
        dirty = _run(returns, dirty_rv)
        scores = [c for c in clean.columns if c != "config_hash"]
        past = clean["target_index"] <= cutoff
        pd.testing.assert_frame_equal(
            clean.loc[past, scores].reset_index(drop=True),
            dirty.loc[dirty["target_index"] <= cutoff, scores].reset_index(drop=True),
            check_exact=True,
        )
        # and the canary can fail: the corrupted region did move
        future = clean["target_index"] > cutoff + 22
        assert not np.array_equal(
            clean.loc[future, "crps"].to_numpy(), dirty.loc[future, "crps"].to_numpy()
        )

    def test_a_bad_origin_costs_one_row_not_the_cell(self) -> None:
        returns, rv = _panel()
        rv = rv.copy()
        rv.iloc[150] = np.nan  # an unusable day inside the backtest range
        out = _run(returns, rv)
        failed = out["missing_reason"] != ""
        assert failed.any() and not failed.all()
        # the origin whose *target* is the bad day is the evaluator's own
        # proxy_nan row; every origin whose 100-day window holds day 150 is a
        # fit_error from validated_rv — validation covers the window the
        # evaluator hands over, as HAR's does, not just the trailing context
        # the model reads; and the cell recovers once the day leaves the window
        reason = out["missing_reason"]
        assert reason[out["target_index"] == 150].tolist() == ["proxy_nan"]
        in_window = (out["origin_index"] >= 150) & (out["origin_index"] <= 249)
        assert reason[in_window].str.contains("fit_error").all()
        assert reason[in_window].str.contains("finite").all()
        assert failed.equals(in_window | (out["target_index"] == 150))
