"""The heavy-model smoke run, driven with weight-free backends so CI covers its wiring.

`volbench.benchmarks.smoke_tsfm` is run by hand on the GPU box; what this
pins is that the module composes the toy series, the splitter, the store and
the summary the same way `benchmarks.toy` does, that the real `models()` set
is constructible without weights, and that the run is deterministic.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
from tsfm_fakes import FakeBackend

from volbench.benchmarks.smoke_tsfm import REFIT_EVERY, models, run_smoke_tsfm
from volbench.benchmarks.toy import ModelEntry
from volbench.models import Chronos, Moirai, PatchTST, TimesFM
from volbench.results import ResultsStore


def _fake_entries() -> list[ModelEntry]:
    fake = FakeBackend()
    return [
        ModelEntry("chronos", functools.partial(Chronos, backend=fake), fits_on_variance=True),
        ModelEntry("timesfm", functools.partial(TimesFM, backend=fake), fits_on_variance=True),
        ModelEntry("moirai", functools.partial(Moirai, backend=fake), fits_on_variance=True),
    ]


def _tiny_patchtst() -> ModelEntry:
    pytest.importorskip("torch")
    factory = functools.partial(
        PatchTST,
        lookback=16,
        patch_len=8,
        stride=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        max_epochs=2,
        patience=2,
        batch_size=16,
        device="cpu",
    )
    return ModelEntry("patchtst", factory, fits_on_variance=True)


class TestModelSet:
    def test_default_set_is_constructible_without_weights(self) -> None:
        # Construction only: a real-backend adapter's `spec()` reads the
        # backend's version and the cached checkpoint's revision, so it needs
        # the `tsfm` extra, which CI never installs.
        labels = [e.label for e in models(device="cpu")]
        assert labels == ["chronos", "timesfm", "moirai", "patchtst"]
        for entry in models(device="cpu"):
            assert entry.fits_on_variance
            assert entry.factory().name

    def test_timegpt_is_opt_in(self) -> None:
        assert "timegpt" not in [e.label for e in models()]
        assert "timegpt" in [e.label for e in models(timegpt=True)]

    def test_refit_cadence_default_is_the_documented_one(self) -> None:
        assert REFIT_EVERY == 21


class TestRun:
    def test_zero_shot_run_lands_in_the_store_and_is_deterministic(self, tmp_path: Path) -> None:
        first = run_smoke_tsfm(out_dir=tmp_path / "a", entries=_fake_entries())
        second = run_smoke_tsfm(out_dir=tmp_path / "b", entries=_fake_entries())
        assert first.n_origins == 200
        assert set(first.results["label"]) == {"chronos", "timesfm", "moirai"}
        assert (first.results["missing_reason"] == "").all()
        assert first.config_hashes == second.config_hashes
        assert len(first.config_hashes) == 3  # three distinct hashes: identity differs
        store = ResultsStore(tmp_path / "a")
        assert set(store.config_hashes()) == set(first.config_hashes.values())
        for label, h in first.config_hashes.items():
            a = (tmp_path / "a" / f"{h}.parquet").read_bytes()
            b = (tmp_path / "b" / f"{h}.parquet").read_bytes()
            assert a == b, label
        assert (tmp_path / "a" / "summary.md").read_text().startswith("# volbench TSFM")

    def test_refit_cadence_changes_no_zero_shot_number(self) -> None:
        daily = run_smoke_tsfm(out_dir=None, entries=_fake_entries(), refit_every=1)
        monthly = run_smoke_tsfm(out_dir=None, entries=_fake_entries(), refit_every=21)
        cols = ["label", "origin_index", "forecast_var", "qlike", "crps"]
        a = daily.results[cols].reset_index(drop=True)
        b = monthly.results[cols].reset_index(drop=True)
        assert a.equals(b)

    def test_patchtst_runs_frozen_between_refits(self, tmp_path: Path) -> None:
        result = run_smoke_tsfm(
            out_dir=tmp_path, entries=[_tiny_patchtst()], window=60, refit_every=50
        )
        rows = result.results
        assert (rows["missing_reason"] == "").all()
        assert (rows["conditioned_through"] == rows["fit_origin"]).all()
