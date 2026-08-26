"""The refit protocol: re-estimate every N origins, re-condition daily between.

M1 report §4.3 recorded that "refit every 21 days" meant "freeze the forecast
for 21 days", because no model implemented ``SupportsUpdate``. Decision taken
on m2/evaluator-hardening: parameters are re-estimated every N origins AND
the conditional state is re-conditioned on every origin's window in between
(``recondition="daily"``, the default). The frozen behaviour stays available
as an explicit ablation arm (``recondition="none"``), never as a default.

Three things are pinned here, separately:

(a) fit count and conditioning index are tracked and asserted apart — at
    refit_every=21 there are ~n/21 fits, yet every row is conditioned through
    its own origin;
(b) at refit_every=1 the setting is a no-op, down to the config hash and the
    parquet bytes — re-conditioning when nothing new was observed must change
    nothing, so any drift there is an implementation bug;
(c) ``recondition="none"`` at refit_every=21 is exactly the old frozen path:
    the forecast issued at the refit origin, held until the next one.

Every train/test index here comes from ``RollingOriginSplitter``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from volbench.benchmarks.toy import (
    ASSET_ID,
    HORIZON,
    SCORING_TARGET,
    SEED,
    STEP,
    WINDOW,
    ModelEntry,
    ToySeries,
    load_series,
    models,
    run_toy_benchmark,
)
from volbench.dist import Distribution
from volbench.evaluate import Recondition, SupportsUpdate, run_backtest
from volbench.models.base import FitDiagnostics, SupportsFitDiagnostics
from volbench.results import ResultsStore, build_config, canonical_repr, config_hash
from volbench.splitter import RollingOriginSplitter

N_ORIGINS = 200

#: The toy benchmark's experiment identities on the committed M2 fixture
#: (refit_every=1). They may change only with a deliberate version, protocol,
#: fixture or target change — never as a side effect of adding a setting that
#: does not bind. History: updated at the shared-target change (m2/cleanup
#: item 1), at the 0.2.0 version bump (item 3 — package_version is in every
#: hash), at the 0.3.0 Phase-2 core integration (D-021: all eight moved with
#: the version; autoets / autoarima / lgbm are new — the classical log-RV
#: models joined the toy benchmark, D-023), at 0.4.0, where two things moved
#: them at once: the version again, and D-018's invalid-target policy, which
#: the four variance-fed cells now carry in ``protocol`` (the *numbers* did
#: not move there — the fixture has no invalid day for compaction to drop,
#: and ``tests/test_compaction.py`` pins that equivalence directly), and at
#: 0.5.0 (D-030), where the version moved all eight again and HAR's move onto
#: the shared smearing retransformation moved **HAR's numbers as well** — its
#: model name is now ``har_rv-smearing`` and its QLIKE 0.1823 -> 0.1806. The
#: other seven cells' numbers are byte-identical to the 0.4.0 run; only their
#: hashes moved, with the version. And at 0.6.0 (D-032), where three things
#: moved them: the version again; the new ``environment`` block, which puts
#: the BLAS thread count in every hash; and — for the two GARCH cells alone —
#: ``fit_tol``, plus ``nu_bounds`` on ``garch11_t``, which are now
#: hyperparameters in ``spec()``. **The two GARCH cells' numbers moved too**,
#: which the other six's did not: bounding nu to (2.1, 50) is a different
#: estimator, and it is what took ``garch11_t``'s BLAS-thread sensitivity from
#: 5.5e-1 to 2.9e-6 (docs/P3_DETERMINISM.md §4).
#:
#: These identities are only defined under the thread pin — hence
#: ``@pytest.mark.pinned_identity``, which skips them on an unpinned shell
#: rather than reporting a machine's different-but-correct answer as a
#: regression.
PINNED_CONFIG_HASHES = {
    "autoarima": "e859c63b865f4ba4dcea064dd109878988234affead5b24521d628cc608aecb9",
    "autoets": "505fc821c85ae49fc16faadc5f87961bfe7b4ac7129caccbcc7355478a7c1c61",
    "ewma": "228a809631262c4ce799c6f2e9f85edebfa7c691b804c70aec5c90b515545d89",
    "garch11": "d5db8995df4c16e8faf3ae1cd75eea463a0f0bedee27a5df4c9aa184f5224464",
    "garch11_t": "dd95b567c1eaddd78bfc01f910d4f61db2bdcef9b83b5832f18a0f4ea06f3075",
    "har": "d12f6e460217bc592fb44dcc923fe30866d4223faaa80ab787a6da7ce5b1060b",
    "lgbm": "adfdb5a13283167a45787ac2e6c438a7cc00ad673a04f982320b9a1d6c7ab701",
    "naive": "e4dfb12b52b6ae5af584a37c9262d819ae9cade6edde4c5ea80182d43163a38c",
}


# --------------------------------------------------------------------------
# wrappers: count calls, or hide the update capability
# --------------------------------------------------------------------------


class CountingFitted:
    def __init__(self, inner: Any, counter: dict[str, int]) -> None:
        self.inner, self.counter = inner, counter

    @property
    def name(self) -> str:
        return str(self.inner.name)

    def spec(self) -> dict[str, Any]:
        return dict(self.inner.spec())

    def predict(self, h: int) -> Distribution:
        dist: Distribution = self.inner.predict(h)
        return dist

    def update(self, train: NDArray[np.float64]) -> CountingFitted:
        self.counter["updates"] += 1
        return CountingFitted(self.inner.update(train), self.counter)


class Counting:
    """Delegates to a real model; counts fits and updates in a shared dict."""

    def __init__(self, inner: Any, counter: dict[str, int]) -> None:
        self.inner, self.counter = inner, counter

    @property
    def name(self) -> str:
        return str(self.inner.name)

    def spec(self) -> dict[str, Any]:
        return dict(self.inner.spec())

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> CountingFitted:
        self.counter["fits"] += 1
        return CountingFitted(self.inner.fit(train), self.counter)


class NoUpdateFitted:
    """A fitted model with the update capability hidden — a pre-M2 model.

    *Only* ``update`` is hidden. Everything else is forwarded, so the
    comparison this stands in for stays a comparison of re-conditioning and
    not of whatever else the wrapper happened not to implement.

    ``fit_diagnostics`` (D-032) is the one forwarded member that cannot be
    left to ``__getattr__``: see :class:`NoUpdateFittedWithDiagnostics`.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in ("update", "inner"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    @property
    def name(self) -> str:
        return str(self.inner.name)

    def spec(self) -> dict[str, Any]:
        return dict(self.inner.spec())

    def predict(self, h: int) -> Distribution:
        dist: Distribution = self.inner.predict(h)
        return dist


class NoUpdateFittedWithDiagnostics(NoUpdateFitted):
    """The same stand-in, for an inner model that reports fit diagnostics.

    ``fit_diagnostics`` is declared on the class rather than reached through
    ``NoUpdateFitted.__getattr__`` because from Python 3.12 a
    ``runtime_checkable`` protocol's ``isinstance`` resolves its members with
    ``inspect.getattr_static`` (CPython gh-102433), which does not consult
    ``__getattr__``. Dynamic forwarding satisfies ``hasattr`` but leaves the
    stand-in invisible to ``SupportsFitDiagnostics`` on 3.12+, which is what
    it was measured doing: ``evaluate._fit_status`` recorded ``""`` on every
    row and ``TestFrozenArm`` compared a real status column against a blank
    one.

    Which class :class:`NoUpdate` uses is decided per fit, so a stand-in for a
    model that reports nothing keeps reporting nothing rather than acquiring a
    capability its inner model never had.
    """

    def fit_diagnostics(self) -> FitDiagnostics:
        diagnostics: FitDiagnostics = self.inner.fit_diagnostics()
        return diagnostics


class NoUpdate:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    @property
    def name(self) -> str:
        return str(self.inner.name)

    def spec(self) -> dict[str, Any]:
        return dict(self.inner.spec())

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> NoUpdateFitted:
        fitted = self.inner.fit(train)
        if isinstance(fitted, SupportsFitDiagnostics):
            return NoUpdateFittedWithDiagnostics(fitted)
        return NoUpdateFitted(fitted)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def toy_backtest(
    entry: ModelEntry,
    model: Any,
    *,
    refit_every: int,
    recondition: Recondition = "daily",
    toy: ToySeries | None = None,
) -> pd.DataFrame:
    returns, proxy, fit_series = (load_series() if toy is None else toy).inputs_for(entry)
    splitter = RollingOriginSplitter(
        window=WINDOW, horizon=HORIZON, step=STEP, refit_every=refit_every
    )
    return run_backtest(
        lambda: model,
        returns,
        proxy,
        splitter,
        SEED,
        asset=ASSET_ID,
        proxy_name=SCORING_TARGET,
        fit_series=fit_series,
        recondition=recondition,
    )


def scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Everything but the experiment identity."""
    return frame.drop(columns=["config_hash"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# (a) fits and conditioning are tracked separately
# --------------------------------------------------------------------------


class TestFitsAndConditioningAreTrackedSeparately:
    @pytest.mark.parametrize("entry", models(), ids=lambda e: e.label)
    def test_refit_every_21_fits_ten_times_but_conditions_through_every_origin(
        self, entry: ModelEntry
    ) -> None:
        counter = {"fits": 0, "updates": 0}
        frame = toy_backtest(entry, Counting(entry.factory(), counter), refit_every=21)

        expected_fits = math.ceil(N_ORIGINS / 21)
        assert len(frame) == N_ORIGINS
        assert counter["fits"] == expected_fits == 10
        assert counter["updates"] == N_ORIGINS - expected_fits == 190
        assert int(frame["refit"].sum()) == expected_fits

        # Parameters: from the block's scheduled refit, never later.
        assert (frame["fit_origin"] <= frame["origin_index"]).all()
        refit_rows = frame[frame["refit"]]
        assert (refit_rows["fit_origin"] == refit_rows["origin_index"]).all()
        assert frame["fit_origin"].nunique() == expected_fits
        # Conditioning: through the row's own origin, on every row.
        assert (frame["conditioned_through"] == frame["origin_index"]).all()
        # And it did something: the forecast moves between refits.
        assert frame["forecast_var"].nunique() > expected_fits
        assert frame["forecast_var"].notna().all()
        assert (frame["missing_reason"] == "").all()

    def test_update_only_ever_sees_observations_at_or_before_the_origin(self) -> None:
        """The leakage half of the contract: whatever ``update`` receives is
        the origin's own splitter window, ending at the origin."""
        seen: list[tuple[int, NDArray[np.float64]]] = []

        class Recorder:
            @property
            def name(self) -> str:
                return "recorder"

            def spec(self) -> dict[str, Any]:
                return {"kind": "recorder"}

            def fit(self, train: NDArray[np.float64], **ctx: Any) -> RecorderFit:
                return RecorderFit(float(np.std(train)))

        class RecorderFit:
            def __init__(self, sigma: float) -> None:
                self.sigma = sigma

            def predict(self, h: int) -> Distribution:
                return Distribution.from_normal(0.0, max(self.sigma, 1e-12))

            def update(self, train: NDArray[np.float64]) -> RecorderFit:
                seen.append((train.size, train.copy()))
                return RecorderFit(float(np.std(train)))

        rng = np.random.default_rng(0)
        series = rng.normal(0.0, 0.01, size=300)
        splitter = RollingOriginSplitter(window=50, horizon=1, step=1, refit_every=7)
        run_backtest(
            Recorder, series, series**2, splitter, 0, asset="SIM", proxy_name="sq"
        )
        origins = [o for o in splitter.split(series.size) if not o.refit]
        assert len(seen) == len(origins)
        for (size, window), origin in zip(seen, origins, strict=True):
            assert size == splitter.window
            assert np.array_equal(window, series[origin.train])
            assert int(origin.train.max()) == origin.origin  # nothing past the cutoff


# --------------------------------------------------------------------------
# (b) equivalence at refit_every=1
# --------------------------------------------------------------------------


class TestEquivalenceAtRefitEveryOne:
    """With one refit per origin there is nothing to re-condition. The setting
    must therefore be invisible: same rows, same config hashes as v0.1.0-m1,
    same parquet bytes. (The gate additionally checks the bytes against a
    fresh run of the pre-change code; see the commit message.)"""

    @pytest.mark.pinned_identity
    def test_daily_and_none_are_byte_identical_and_keep_the_pinned_hashes(
        self, tmp_path: Path
    ) -> None:
        daily = run_toy_benchmark(out_dir=tmp_path / "daily", refit_every=1, recondition="daily")
        frozen = run_toy_benchmark(out_dir=tmp_path / "none", refit_every=1, recondition="none")

        # On a mismatch, show the whole canonical config of one cell: whether
        # the model spec, the data digests, the splitter or the version moved
        # is the first thing anyone needs to know, and the hash alone hides it.
        naive_config = ResultsStore(tmp_path / "daily").read_config(daily.config_hashes["naive"])
        assert daily.config_hashes == frozen.config_hashes == PINNED_CONFIG_HASHES, (
            "pinned identities moved; the naive cell's config on this machine is:\n"
            + canonical_repr(naive_config)
        )
        for label, digest in daily.config_hashes.items():
            a = (tmp_path / "daily" / f"{digest}.parquet").read_bytes()
            b = (tmp_path / "none" / f"{digest}.parquet").read_bytes()
            assert a == b, label
        pd.testing.assert_frame_equal(daily.results, frozen.results)

    def test_the_setting_is_not_recorded_when_it_cannot_bind(self) -> None:
        entry = next(e for e in models() if e.label == "ewma")
        one = toy_backtest(entry, entry.factory(), refit_every=1, recondition="none")
        assert "protocol" not in one.attrs["config"]
        many = toy_backtest(entry, entry.factory(), refit_every=21, recondition="none")
        assert many.attrs["config"]["protocol"] == {"recondition": "none"}


# --------------------------------------------------------------------------
# (c) the frozen ablation arm
# --------------------------------------------------------------------------


class TestFrozenArm:
    @pytest.mark.parametrize("entry", models(), ids=lambda e: e.label)
    def test_none_holds_the_refit_forecast_for_the_whole_block(self, entry: ModelEntry) -> None:
        frame = toy_backtest(entry, entry.factory(), refit_every=21, recondition="none")
        assert (frame["conditioned_through"] == frame["fit_origin"]).all()
        per_block = frame.groupby("fit_origin")["forecast_var"].nunique()
        assert (per_block == 1).all()
        assert frame["fit_origin"].nunique() == 10

    @pytest.mark.parametrize("entry", models(), ids=lambda e: e.label)
    def test_none_reproduces_a_model_without_update_under_daily(self, entry: ModelEntry) -> None:
        """The old frozen numbers, operationally: a model that cannot
        re-condition, run under the default, must land on exactly the rows the
        ablation arm produces for a model that can."""
        frozen = toy_backtest(entry, entry.factory(), refit_every=21, recondition="none")
        legacy = toy_backtest(entry, NoUpdate(entry.factory()), refit_every=21, recondition="daily")
        pd.testing.assert_frame_equal(scores(frozen), scores(legacy))

    def test_daily_and_none_differ_at_refit_every_21(self) -> None:
        entry = next(e for e in models() if e.label == "garch11")
        daily = toy_backtest(entry, entry.factory(), refit_every=21, recondition="daily")
        frozen = toy_backtest(entry, entry.factory(), refit_every=21, recondition="none")
        assert daily.attrs["config_hash"] != frozen.attrs["config_hash"]
        assert not np.allclose(daily["forecast_var"], frozen["forecast_var"])
        # Same parameters at the refit origins, so those rows agree exactly.
        at_refits = daily["refit"]
        assert np.array_equal(
            daily.loc[at_refits, "forecast_var"].to_numpy(),
            frozen.loc[at_refits, "forecast_var"].to_numpy(),
        )


class TestTheStandInHidesOnlyWhatItClaimsTo:
    """``NoUpdate`` is a control only if it hides ``update`` and nothing else.

    Regression for a defect that reached ``main``: ``NoUpdateFitted`` forwarded
    ``fit_diagnostics`` through ``__getattr__`` alone, which ``hasattr``
    accepts but the 3.12+ ``runtime_checkable`` ``isinstance`` does not
    (CPython gh-102433 resolves protocol members with
    ``inspect.getattr_static``). ``TestFrozenArm`` above then compared a real
    ``fit_status`` column against a blank one and failed on 3.12/3.13 while
    passing on 3.11. Asserting the capabilities themselves turns the next such
    change into a named one-line failure rather than a 100%-different column,
    and it costs one fit per model instead of two backtests.
    """

    @pytest.mark.parametrize("entry", models(), ids=lambda e: e.label)
    def test_update_is_hidden_and_the_other_capabilities_are_not(
        self, entry: ModelEntry
    ) -> None:
        toy = load_series()
        source = toy.targets[SCORING_TARGET] if entry.fits_on_variance else toy.returns
        train = source.to_numpy(dtype=np.float64)[:WINDOW]

        inner = entry.factory().fit(train)
        stand_in = NoUpdate(entry.factory()).fit(train)

        # The one capability it exists to hide.
        assert isinstance(inner, SupportsUpdate)
        assert not isinstance(stand_in, SupportsUpdate)

        # Every other one it claims to forward, by the same test the evaluator
        # applies — `isinstance`, not `hasattr`.
        assert isinstance(stand_in, SupportsFitDiagnostics) == isinstance(
            inner, SupportsFitDiagnostics
        )
        if isinstance(inner, SupportsFitDiagnostics):
            assert stand_in.fit_diagnostics().status() == inner.fit_diagnostics().status()
        assert stand_in.name == inner.name
        assert stand_in.spec() == inner.spec()


# --------------------------------------------------------------------------
# the leakage canary, on the re-conditioning path
# --------------------------------------------------------------------------


class TestLeakageCanaryUnderDailyReconditioning:
    """`.claude/skills/leakage-check`'s demanded canary, aimed at ``update``.

    Neither existing canary reaches this path: ``test_evaluate``'s uses a
    model without ``update``, and the M1 smoke canary runs at refit_every=1
    where ``update`` is unreachable. Re-conditioning is exactly where a leak
    would live, so: corrupt everything strictly after a cutoff, run at
    refit_every=21 under ``recondition="daily"``, and require every row whose
    target is at or before the cutoff — most of them re-conditioned, not
    refitted — to be bit-identical to the clean run.
    """

    @pytest.mark.parametrize("entry", models(), ids=lambda e: e.label)
    def test_future_corruption_cannot_change_reconditioned_past_forecasts(
        self, entry: ModelEntry
    ) -> None:
        toy = load_series()
        cutoff = WINDOW + 60  # three refit blocks, 58 re-conditioned origins, all clean
        rng = np.random.default_rng(99)
        n_tail = toy.returns.size - (cutoff + 1)
        bad_returns = toy.returns.copy()
        bad_returns.iloc[cutoff + 1 :] = rng.normal(0.0, 0.5, size=n_tail)
        bad_targets = {}
        for name, target in toy.targets.items():
            corrupted = target.copy()
            corrupted.iloc[cutoff + 1 :] = np.exp(rng.normal(0.0, 1.0, size=n_tail))
            bad_targets[name] = corrupted

        clean = toy_backtest(entry, entry.factory(), refit_every=21, toy=toy)
        dirty = toy_backtest(
            entry,
            entry.factory(),
            refit_every=21,
            toy=ToySeries(returns=bad_returns, targets=bad_targets),
        )

        before = clean[clean["target_index"] <= cutoff]
        after = dirty[dirty["target_index"] <= cutoff]
        assert len(before) == 61
        assert int((~before["refit"]).sum()) == 58  # the update path, not the fit path
        assert (before["conditioned_through"] == before["origin_index"]).all()
        pd.testing.assert_frame_equal(scores(before), scores(after), check_exact=True)

        # A canary that cannot die proves nothing: the poison must have reached
        # every forecast whose *window* contains corrupted data — origins past
        # the cutoff. (The origin at the cutoff has a clean window and a
        # corrupted target, so its forecast is rightly unchanged.)
        poisoned = clean["origin_index"] > cutoff
        later_clean = clean.loc[poisoned, "forecast_var"].to_numpy()
        later_dirty = dirty.loc[poisoned, "forecast_var"].to_numpy()
        assert len(later_clean) > 100
        assert (later_clean != later_dirty).all()


# --------------------------------------------------------------------------
# the hash rule, in isolation
# --------------------------------------------------------------------------


class TestHashRule:
    def test_protocol_enters_the_hash_only_when_given(self) -> None:
        splitter = RollingOriginSplitter(window=10, refit_every=21)
        common: dict[str, Any] = {
            "model_name": "m",
            "model_spec": {},
            "data_spec": {},
            "splitter": splitter,
            "seed": 0,
            "version": "0.1.0",
        }
        bare = build_config(**common)
        assert "protocol" not in bare
        assert build_config(protocol=None, **common) == bare
        assert build_config(protocol={}, **common) == bare
        daily = build_config(protocol={"recondition": "daily"}, **common)
        frozen = build_config(protocol={"recondition": "none"}, **common)
        assert daily["protocol"] == {"recondition": "daily"}
        assert len({config_hash(bare), config_hash(daily), config_hash(frozen)}) == 3

    def test_an_unknown_setting_is_rejected(self) -> None:
        entry = models()[0]
        with pytest.raises(ValueError, match="recondition must be"):
            toy_backtest(entry, entry.factory(), refit_every=5, recondition="weekly")  # type: ignore[arg-type]
