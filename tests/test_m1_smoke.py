"""M1 end-to-end smoke test: all three Phase 1 streams, one series, one table.

This is the milestone's acceptance test (docs/phase1_prompts.md stream D,
task 4). Every other test file exercises one stream in isolation; this one is
the only place where data ingestion, the model adapters and the evaluator run
against each other, so it is where an integration regression will show up
first.

What it pins:

- the wiring produces a full scored table — four baselines, ~200 rolling
  origins, no silently dropped rows;
- the run is deterministic under a fixed seed, down to the parquet bytes
  (CLAUDE.md rule 3, and the claim D-011 rests on);
- the committed fixture is itself reproducible from its generator, so
  `make reproduce` rebuilds the benchmark rather than re-reading a file
  nobody can regenerate;
- temporal integrity survives the composition: every forecast's target lands
  strictly after the origin it was issued at, and no model was fitted on data
  past its own origin.

Runtime budget: the brief allows two minutes. It runs in a few seconds; the
assertion is kept at the brief's bound so it fails loudly if some future
change makes the toy benchmark expensive, without being flaky on a slow CI
runner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volbench.benchmarks.make_toy_asset import DEFAULT_PATH, simulate_ohlc
from volbench.benchmarks.toy import ASSET_ID, WINDOW, ToyBenchmarkResult, models, run_toy_benchmark
from volbench.results import ResultsStore

RUNTIME_BUDGET_SECONDS = 120.0
EXPECTED_ORIGINS = 200
EXPECTED_MODELS = 4


@dataclass(frozen=True, eq=False)
class TimedRun:
    """One benchmark run plus what the test needs to judge it."""

    result: ToyBenchmarkResult
    elapsed: float
    out_dir: Path

    @property
    def results(self) -> pd.DataFrame:
        return self.result.results


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory: pytest.TempPathFactory) -> TimedRun:
    """Run the benchmark once for the whole module — it is the expensive part."""
    out = tmp_path_factory.mktemp("toy_benchmark")
    started = time.perf_counter()
    result = run_toy_benchmark(out_dir=out)
    return TimedRun(result=result, elapsed=time.perf_counter() - started, out_dir=out)


class TestFixtureIsReproducible:
    def test_committed_fixture_matches_its_generator(self) -> None:
        """`make reproduce` rebuilds the fixture; if the committed file and the
        generator disagree, the benchmark is not reproducible from scratch."""
        regenerated = simulate_ohlc()
        committed = pd.read_csv(DEFAULT_PATH, parse_dates=["date"])
        committed["date"] = committed["date"].dt.tz_localize("UTC")
        pd.testing.assert_frame_equal(regenerated, committed, check_exact=False, rtol=1e-12)

    def test_fixture_bars_are_internally_consistent(self) -> None:
        bars = pd.read_csv(DEFAULT_PATH)
        assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
        assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
        # A zero-range day would make the Parkinson proxy 0, which HAR-RV takes
        # the log of. See docs/M1_REPORT.md risk 2.
        assert (bars["high"] > bars["low"]).all()


class TestScoredTable:
    def test_shape_is_every_model_at_every_origin(self, benchmark: TimedRun) -> None:
        assert benchmark.result.n_origins == EXPECTED_ORIGINS
        assert len(benchmark.results) == EXPECTED_ORIGINS * EXPECTED_MODELS
        assert set(benchmark.results["label"]) == {e.label for e in models()}

    def test_nothing_was_silently_dropped(self, benchmark: TimedRun) -> None:
        frame = benchmark.results
        assert (frame["missing_reason"] == "").all(), frame["missing_reason"].unique()
        for column in ("crps", "log_score", "qlike", "forecast_var", "realized_return"):
            assert frame[column].notna().all(), f"{column} has gaps"
            assert np.isfinite(frame[column]).all(), f"{column} has non-finite values"

    def test_forecasts_are_daily_variance_not_annualized(self, benchmark: TimedRun) -> None:
        frame = benchmark.results
        assert (frame["forecast_var"] > 0.0).all()
        # A daily equity-index return variance is ~1e-4. Annualizing would put
        # it near 3e-2 — two orders out, and caught here (CLAUDE.md rule 2).
        assert frame["forecast_var"].max() < 1e-2

    def test_summary_ranks_every_model_on_every_score(self, benchmark: TimedRun) -> None:
        summary = benchmark.result.summary
        assert len(summary) == EXPECTED_MODELS
        assert (summary["n_scored"] == EXPECTED_ORIGINS).all()
        for column in ("crps", "log_score", "qlike", "pinball_0p01", "hitrate_0p01"):
            assert summary[column].notna().all()

    def test_summary_files_were_written(self, benchmark: TimedRun) -> None:
        out = benchmark.out_dir
        assert (out / "summary.csv").is_file()
        assert (out / "summary.md").is_file()
        assert len(list(out.glob("*.parquet"))) == EXPECTED_MODELS


class TestTemporalIntegrity:
    """The composition must not lose what the splitter guarantees alone."""

    def test_every_target_lands_after_its_origin(self, benchmark: TimedRun) -> None:
        frame = benchmark.results
        assert (frame["target_index"] > frame["origin_index"]).all()
        assert (frame["target_index"] == frame["origin_index"] + frame["horizon"]).all()

    def test_no_model_was_fitted_on_data_past_its_own_origin(self, benchmark: TimedRun) -> None:
        frame = benchmark.results
        assert (frame["fit_origin"] <= frame["origin_index"]).all()
        assert (frame["conditioned_through"] <= frame["origin_index"]).all()

    def test_first_origin_is_the_first_full_window(self, benchmark: TimedRun) -> None:
        frame = benchmark.results
        assert int(frame["origin_index"].min()) == WINDOW - 1


class TestDeterminism:
    """The repo's headline reproducibility claim, at the integration level."""

    def test_two_runs_produce_byte_identical_parquet(self, tmp_path: Path) -> None:
        first = run_toy_benchmark(out_dir=tmp_path / "a")
        second = run_toy_benchmark(out_dir=tmp_path / "b")

        assert first.config_hashes == second.config_hashes
        for label, hash_value in first.config_hashes.items():
            a = (tmp_path / "a" / f"{hash_value}.parquet").read_bytes()
            b = (tmp_path / "b" / f"{hash_value}.parquet").read_bytes()
            assert a == b, f"{label} parquet differs between identical runs"

    def test_rerunning_against_a_populated_store_short_circuits(self, tmp_path: Path) -> None:
        run_toy_benchmark(out_dir=tmp_path)
        store = ResultsStore(tmp_path)
        before = {h: store.fragment_path(h).stat().st_mtime_ns for h in store.config_hashes()}

        again = run_toy_benchmark(out_dir=tmp_path)

        after = {h: store.fragment_path(h).stat().st_mtime_ns for h in store.config_hashes()}
        assert before == after, "a cached run rewrote its fragments"
        assert set(again.config_hashes.values()) == set(before)


class TestRuntimeBudget:
    def test_runs_inside_the_two_minute_budget(self, benchmark: TimedRun) -> None:
        assert benchmark.elapsed < RUNTIME_BUDGET_SECONDS, (
            f"toy benchmark took {benchmark.elapsed:.1f}s"
        )


class TestProvenance:
    def test_every_row_carries_seed_and_config_hash(self, benchmark: TimedRun) -> None:
        frame = benchmark.results
        assert (frame["seed"] > 0).all()
        assert frame["config_hash"].str.len().eq(64).all()
        assert (frame["asset"] == ASSET_ID).all()
        # One hash per model, and models must not collide onto one hash.
        assert frame["config_hash"].nunique() == EXPECTED_MODELS


class TestLeakageCanary:
    """The leakage-check skill's demanded canary, at the integration level.

    Corrupt every observation strictly after a cutoff T, rerun the whole
    benchmark, and require that every forecast whose target lands at or before
    T is *bit-identical* to the uncorrupted run. Anything that reads ahead —
    a transform fitted on the full series, an off-by-one at the train/test
    join, a proxy that peeks at t+1 — changes those numbers and fails here.

    This is stronger than the per-stream leakage tests: it can only pass if
    the data layer, the models and the evaluator are *jointly* backward-
    looking. The composition is what M1 added, so the composition is what
    needs the canary.
    """

    @staticmethod
    def _run(returns: np.ndarray, proxy: np.ndarray) -> pd.DataFrame:
        from volbench.benchmarks.toy import ASSET_ID, HORIZON, PROXY_NAME, SEED, STEP
        from volbench.evaluate import run_backtest
        from volbench.splitter import RollingOriginSplitter

        splitter = RollingOriginSplitter(
            window=WINDOW, horizon=HORIZON, step=STEP, refit_every=1
        )
        frames = [
            run_backtest(
                entry.factory,
                returns,
                proxy,
                splitter,
                SEED,
                asset=ASSET_ID,
                proxy_name=PROXY_NAME,
                fit_series=proxy if entry.fits_on_variance else None,
            ).assign(label=entry.label)
            for entry in models()
        ]
        return pd.concat(frames, ignore_index=True)

    def test_future_corruption_cannot_change_past_forecasts(self) -> None:
        from volbench.benchmarks.toy import load_series

        returns, proxy = load_series()
        cutoff = WINDOW + 50  # deep enough in that ~50 origins are fully clean

        corrupted_returns = returns.copy()
        corrupted_proxy = proxy.copy()
        rng = np.random.default_rng(999)
        n_tail = returns.size - (cutoff + 1)
        corrupted_returns.iloc[cutoff + 1 :] = rng.normal(0.0, 0.5, size=n_tail)
        corrupted_proxy.iloc[cutoff + 1 :] = np.exp(rng.normal(0.0, 1.0, size=n_tail))

        clean = self._run(returns, proxy)
        dirty = self._run(corrupted_returns, corrupted_proxy)

        # `config_hash` legitimately differs — the data digest changed — so
        # compare the scored numbers, not the provenance columns.
        scores = [c for c in clean.columns if c != "config_hash"]
        clean_past = clean[clean["target_index"] <= cutoff][scores].reset_index(drop=True)
        dirty_past = dirty[dirty["target_index"] <= cutoff][scores].reset_index(drop=True)

        assert len(clean_past) > 0, "cutoff left no fully-clean origins to compare"
        pd.testing.assert_frame_equal(clean_past, dirty_past, check_exact=True)

    def test_the_canary_can_actually_fail(self) -> None:
        """Corrupting the past *must* move the same numbers the future did not.

        Without this, a canary that compared two empty frames — or two runs
        that ignored the data entirely — would pass and prove nothing.
        """
        from volbench.benchmarks.toy import load_series

        returns, proxy = load_series()
        cutoff = WINDOW + 50

        corrupted_returns = returns.copy()
        rng = np.random.default_rng(1234)
        corrupted_returns.iloc[: cutoff + 1] = rng.normal(0.0, 0.5, size=cutoff + 1)

        clean = self._run(returns, proxy)
        dirty = self._run(corrupted_returns, proxy)

        clean_past = clean[clean["target_index"] <= cutoff]["crps"].reset_index(drop=True)
        dirty_past = dirty[dirty["target_index"] <= cutoff]["crps"].reset_index(drop=True)
        assert not np.allclose(clean_past, dirty_past)
