"""The three ablation flags on the primary-grid driver, and what they move.

D-034 schedules three arms before the results freeze — D-019's window-1000
robustness arm, D-015's frozen-forecast control (``recondition="none"``) and
D-017's labelled robustness targets — and the driver could express none of
them. These tests are the contract of the flags that express them now.

**Both halves of every flag.** A flag that always moves the config hash is as
broken as one that never does: the first would fragment a store into cells
nothing can find again, the second would serve one arm's fragments for
another's. So each flag is checked twice — the hash moves when the setting
moves, and the hash is *unchanged* when the flag is passed its own default.
The second half is what pins "the defaults are the headline protocol", which
is the property that lets an arm run share a store with the primary grid at
all.

The runs here are on a synthetic panel: ``build_panel`` reads hand-downloaded
archives under the gitignored ``data/raw`` (docs/data_licenses.md), so it is
absent in CI and in a clean checkout. Everything the flags touch — the arm,
the splitter, the proxy, the config hash — is exercised in full regardless,
because a config hash is a function of the series' *contents* and not of
where they came from. The one thing only real data can check, that a default
run reproduces the committed grid's hashes cell for cell, is a run rather
than a test (``build_panel`` alone is ~65 s): docs/P3_ABLATION_ARMS.md §2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from volbench.benchmarks import grid_primary
from volbench.benchmarks.grid_primary import ARM, asset_data, protocol_arm
from volbench.data.panel import (
    TARGET_NAMES,
    BarQuality,
    PanelSeries,
    build_targets,
)
from volbench.data.types import TimeSeriesFrame
from volbench.execute import SerialExecutor
from volbench.models import HAR, NaiveVol
from volbench.runner import ModelConfig

#: Long enough for the window-1000 arm to have origins at all, which is the
#: setting the flag exists for; at window 500 the same series gives 749.
N_BARS = 1_250

#: The two cheap configs the flag tests run: one return-fed, one variance-fed.
#: The pair matters — ``--target`` must move the *proxy* on both and the *fit
#: series* on neither (D-017), and only a variance-fed cell has a fit series
#: whose digest could have moved.
CONFIGS = [ModelConfig("naive", NaiveVol), ModelConfig("har", HAR, fits_on_variance=True)]

_INTRADAY_STEPS = 24


def _synthetic_series(asset_id: str = "TOY", *, seed: int = 7) -> PanelSeries:
    """One panel series from a simulated intraday path, with all four targets.

    The generator is the one ``tests/test_data_diagnostics.py`` uses: an
    overnight jump plus an intraday random walk, so open/high/low/close are a
    consistent bar and ``build_targets`` produces four finite, strictly
    positive variance targets rather than the NaNs a hand-written frame would.
    """
    index = pd.bdate_range("2015-01-01", periods=N_BARS, tz="UTC")
    rng = np.random.default_rng(seed)
    step = 0.01 / np.sqrt(_INTRADAY_STEPS)
    jumps = rng.normal(0.0, 0.004, size=N_BARS)
    path = np.cumsum(rng.normal(0.0, step, size=(N_BARS, _INTRADAY_STEPS)), axis=1)

    log_open = np.empty(N_BARS)
    log_price = np.log(100.0)
    for i in range(N_BARS):
        log_open[i] = log_price + jumps[i]
        log_price = log_open[i] + path[i, -1]

    data = pd.DataFrame(
        {
            "open": np.exp(log_open),
            "high": np.exp(log_open + np.maximum(path.max(axis=1), 0.0)),
            "low": np.exp(log_open + np.minimum(path.min(axis=1), 0.0)),
            "close": np.exp(log_open + path[:, -1]),
        },
        index=index,
    )
    targets, components = build_targets(data)
    return PanelSeries(
        asset_id=asset_id,
        source="test",
        role="index",
        description="synthetic",
        frame=TimeSeriesFrame(data=data, asset_id=asset_id, source="test"),
        targets=targets,
        components=components,
        quality=BarQuality(
            n_bars=N_BARS, repaired=0, inconsistent=0, zero_range=0, non_positive=0
        ),
        primary_target="overnight_plus_range",
        archive_start=index[0],
        archive_end=index[-1],
    )


@pytest.fixture(scope="module")
def series() -> PanelSeries:
    return _synthetic_series()


class Run:
    """One driver run's manifest and store, as the flag tests read them."""

    def __init__(self, manifest: dict[str, Any], store: Path) -> None:
        self.manifest = manifest
        self.store = store

    @property
    def hashes(self) -> dict[str, str]:
        """``"asset/model" -> config_hash`` over every cell of the run."""
        return {f"{c['asset']}/{c['model']}": c["config_hash"] for c in self.manifest["cells"]}

    @property
    def arms(self) -> set[str]:
        return {c["arm"] for c in self.manifest["cells"]}

    def config(self, cell: str) -> dict[str, Any]:
        """The config sidecar the store wrote for one cell — what was hashed."""
        payload: dict[str, Any] = json.loads(
            (self.store / f"{self.hashes[cell]}.json").read_text(encoding="utf-8")
        )
        return payload


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, series: PanelSeries):
    """Run the driver's ``main`` on the synthetic panel and read what it wrote.

    Everything is redirected into ``tmp_path``: the store, the reports and —
    the one that would matter — the manifest, which under the default tag is
    the committed ``docs/P3_GRID_manifest.json``. A tag of its own is passed
    as well, so a test can never write the study's manifest even if a default
    changed underneath it.
    """
    monkeypatch.setattr(grid_primary, "build_panel", lambda: {series.asset_id: series})
    monkeypatch.setattr(grid_primary, "model_configs", lambda **_: list(CONFIGS))
    # The lanes' backends: a pool per cell would dominate the runtime of a
    # test whose subject is a hash, and SerialExecutor is what run_grid
    # defaults to anyway.
    monkeypatch.setattr(grid_primary, "ProcessExecutor", lambda **_: SerialExecutor())

    def _run(*flags: str, tag: str = "arm_test") -> Run:
        out = tmp_path / tag
        code = grid_primary.main(
            [
                "--out-dir", str(out),
                "--manifest-dir", str(out),
                "--tag", tag,
                "--archive-dir", str(tmp_path / "archive"),
                # Named explicitly because the ``--target`` guard reads the
                # panel's *declared* membership (CRYPTO_PANEL) rather than the
                # loaded panel, so that it can refuse before a minute of
                # archive reading. A synthetic panel has to say who it is.
                "--assets", series.asset_id,
                *flags,
            ]
        )
        assert code == 0
        manifest = json.loads(
            (out / grid_primary.manifest_name(tag)).read_text(encoding="utf-8")
        )
        return Run(manifest, out / "store")

    return _run


class TestTheArmObject:
    """``protocol_arm`` is the only place the flags become an arm."""

    def test_no_flag_is_the_headline_arm(self) -> None:
        """The half that makes an arm run comparable to the primary grid at
        all: the defaults are not *like* the headline protocol, they are it."""
        assert protocol_arm() == ARM

    def test_it_carries_every_headline_setting_the_flags_do_not_move(self) -> None:
        arm = protocol_arm(window=1000, recondition="none")
        assert (arm.refit_every, arm.step) == (ARM.refit_every, ARM.step)
        assert arm.invalid_target_policy == ARM.invalid_target_policy

    @pytest.mark.parametrize(
        ("kwargs", "label"),
        [
            ({}, "headline"),
            ({"window": 1000}, "w1000"),
            ({"recondition": "none"}, "recondition-none"),
            ({"target": "parkinson"}, "target-parkinson"),
            ({"window": 1000, "recondition": "none"}, "w1000+recondition-none"),
        ],
    )
    def test_the_label_names_what_moved(self, kwargs: dict[str, Any], label: str) -> None:
        """The label is the manifest's per-cell handle for the arm and is the
        only field free to carry the scoring target, which is not a field of
        the arm because it is a property of the cell (D-017)."""
        assert protocol_arm(**kwargs).label == label


class TestTheWindowFlag:
    """D-019's robustness arm. A splitter field, so it is in every hash."""

    def test_a_different_window_moves_every_cell(self, run: Any) -> None:
        headline = run(tag="w500").hashes
        robustness = run("--window", "1000", tag="w1000").hashes
        assert set(headline) == set(robustness)
        assert not (set(headline.values()) & set(robustness.values()))

    def test_the_default_window_moves_nothing(self, run: Any) -> None:
        """Passing the flag its own value must be indistinguishable from not
        passing it, or every arm run would fragment the store."""
        assert run(tag="w_none").hashes == run("--window", str(ARM.window), tag="w_500").hashes

    def test_it_is_the_splitter_that_carries_it(self, run: Any) -> None:
        config = run("--window", "1000", tag="w1000_cfg").config("TOY/naive")
        assert config["splitter"]["window"] == 1000

    def test_the_arms_do_not_score_the_same_origins(self, run: Any) -> None:
        """The consequence D-019 has to be compared under: a window-1000 cell
        has 500 fewer origins than its window-500 counterpart, so a
        window-sensitivity comparison runs on the intersection of the origins
        both scored or reports coverage per arm (docs/P3_ABLATION_ARMS.md)."""
        rows = {
            c["asset"] + "/" + c["model"]: c["n_rows"]
            for c in run(tag="w500_rows").manifest["cells"]
        }
        long_rows = {
            c["asset"] + "/" + c["model"]: c["n_rows"]
            for c in run("--window", "1000", tag="w1000_rows").manifest["cells"]
        }
        assert all(long_rows[cell] == rows[cell] - 500 for cell in rows)


class TestTheReconditionFlag:
    """D-015's ablation arm: the forecast frozen between refits."""

    def test_recondition_none_moves_every_cell(self, run: Any) -> None:
        daily = run(tag="rc_daily").hashes
        frozen = run("--recondition", "none", tag="rc_none").hashes
        assert set(daily) == set(frozen)
        assert not (set(daily.values()) & set(frozen.values()))

    def test_the_default_recondition_moves_nothing(self, run: Any) -> None:
        assert (
            run(tag="rc_bare").hashes
            == run("--recondition", ARM.recondition, tag="rc_flagged").hashes
        )

    def test_it_is_the_protocol_block_that_carries_it(self, run: Any) -> None:
        """``recondition`` is hashed under ``protocol`` and only when it can
        change a number, i.e. when ``refit_every > 1`` (D-015). The arm's
        cadence is 21, so it binds on every cell of every arm run here."""
        config = run("--recondition", "none", tag="rc_cfg").config("TOY/har")
        assert config["protocol"]["recondition"] == "none"
        assert config["splitter"]["refit_every"] > 1


class TestTheTargetFlag:
    """D-017's labelled robustness proxy: re-scores, never re-fits."""

    def test_a_different_target_moves_every_cell(self, run: Any) -> None:
        primary = run(tag="t_primary").hashes
        parkinson = run("--target", "parkinson", tag="t_parkinson").hashes
        assert set(primary) == set(parkinson)
        assert not (set(primary.values()) & set(parkinson.values()))

    def test_the_assets_own_primary_moves_nothing(self, run: Any) -> None:
        """The synthetic asset's primary is ``overnight_plus_range`` (D-016),
        so naming it explicitly must reproduce the default run exactly."""
        assert (
            run(tag="t_bare").hashes
            == run("--target", "overnight_plus_range", tag="t_named").hashes
        )

    @pytest.mark.parametrize("target", [t for t in TARGET_NAMES if t != "overnight_plus_range"])
    def test_every_robustness_target_is_a_different_experiment(
        self, run: Any, target: str
    ) -> None:
        primary = set(run(tag=f"t_base_{target}").hashes.values())
        assert not (primary & set(run("--target", target, tag=f"t_{target}").hashes.values()))

    def test_it_moves_the_proxy_and_not_the_fit_series(self, run: Any) -> None:
        """D-017 in one assertion, on a variance-fed cell: what a model is fed
        is a modelling contract and does not move with the evaluation. If this
        ever fails, the arm is no longer re-scoring the same forecasts and its
        comparison with the primary means nothing."""
        primary = run(tag="t_fit_primary").config("TOY/har")
        parkinson = run("--target", "parkinson", tag="t_fit_parkinson").config("TOY/har")

        assert primary["data"]["proxy"]["name"] == "overnight_plus_range"
        assert parkinson["data"]["proxy"]["name"] == "parkinson"
        assert primary["data"]["proxy"]["sha256"] != parkinson["data"]["proxy"]["sha256"]
        assert primary["data"]["fit_series_sha256"] == parkinson["data"]["fit_series_sha256"]
        assert primary["data"]["series_sha256"] == parkinson["data"]["series_sha256"]

    def test_the_arm_label_records_which_target_was_scored(self, run: Any) -> None:
        """A committed manifest has to say what its cells were scored against;
        the arm label is where it says it."""
        assert run("--target", "garman_klass", tag="t_label").arms == {"target-garman_klass"}


class TestAssetData:
    """The seam ``--target`` acts at, checked without running a grid."""

    def test_the_default_is_the_assets_own_primary(self, series: PanelSeries) -> None:
        data = asset_data(series)
        assert data.proxy_name == series.primary_target
        pd.testing.assert_series_equal(data.proxy, data.variance)

    def test_a_named_target_moves_the_proxy_only(self, series: PanelSeries) -> None:
        data = asset_data(series, "garman_klass")
        assert data.proxy_name == "garman_klass"
        pd.testing.assert_series_equal(
            data.variance, asset_data(series).variance, check_names=False
        )
        assert not data.proxy.equals(data.variance)

    def test_the_three_series_stay_on_one_calendar(self, series: PanelSeries) -> None:
        data = asset_data(series, "squared_return")
        assert data.variance is not None
        assert data.returns.index.equals(data.proxy.index)
        assert data.returns.index.equals(data.variance.index)

    def test_an_unknown_target_is_refused(self, series: PanelSeries) -> None:
        with pytest.raises(ValueError, match="unknown target"):
            asset_data(series, "realized_variance")


class TestTheCryptoGuard:
    """``--target`` names a range estimator; crypto is not scored on one.

    D-004 and ``build_crypto_series``: on a 24/7 market the "overnight" term
    of a close-to-close estimator is a one-minute gap, and the range targets
    computed there are diagnostics. The robustness-target arm is therefore
    the nine equity assets, and asking for anything else is refused rather
    than quietly scored — before the panel is built, so the refusal costs a
    second and not a minute.
    """

    def test_it_refuses_a_whole_panel_target_run(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            grid_primary.main(["--target", "parkinson", "--tag", "guard"])
        message = capsys.readouterr().err
        assert "BTC-USD" in message and "ETH-USD" in message
        assert "D-004" in message and "--assets" in message

    def test_it_refuses_a_run_that_names_a_crypto_asset(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            grid_primary.main(
                ["--target", "parkinson", "--assets", "SPY", "BTC-USD", "--tag", "guard"]
            )
        assert "BTC-USD" in capsys.readouterr().err

    def test_it_lets_an_equity_only_target_run_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, series: PanelSeries
    ) -> None:
        """The other half: a guard that fires on everything gets switched off.
        The nine equity assets are exactly what the arm runs on."""
        spy = _synthetic_series("SPY")
        monkeypatch.setattr(grid_primary, "build_panel", lambda: {"SPY": spy})
        monkeypatch.setattr(grid_primary, "model_configs", lambda **_: [CONFIGS[0]])
        monkeypatch.setattr(grid_primary, "ProcessExecutor", lambda **_: SerialExecutor())
        code = grid_primary.main(
            [
                "--target", "parkinson",
                "--assets", "SPY",
                "--out-dir", str(tmp_path),
                "--manifest-dir", str(tmp_path),
                "--archive-dir", str(tmp_path / "archive"),
                "--tag", "equity_only",
            ]
        )
        assert code == 0
