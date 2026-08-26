"""D-032: the BLAS thread count is part of a run's identity, and fits say how they went.

Two defects closed here, and they are different in kind.

The first is a *reproducibility* one. Threaded OpenBLAS reorders a reduction by
an ulp; ``arch``'s SLSQP amplifies that into a different local optimum of the
Student-t GARCH likelihood. Unlike the numpy kernel family (D-026), the thread
count moves no content digest, so a 32-thread fragment and a 1-thread fragment
of one cell shared a ``config_hash`` and the store was free to serve either for
the other. The tests below pin the disjunction that has to hold instead: two
thread counts produce **either** identical numbers **or** different hashes.

The second is an *observability* one. The GARCH adapter degrades to EWMA rather
than raising, which is right, but before D-032 it did so invisibly: a cell that
ran EWMA on 40 of its 200 origins scored exactly like one that ran none. The
status is now on every row and counted on every manifest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from volbench.determinism import (
    KERNEL_PIN_VAR,
    PINNED_THREADS,
    THREAD_PIN_VARS,
    blas_info,
    determinism_env,
    environment_report,
    environment_spec,
    is_pinned,
    thread_pin,
)
from volbench.models.base import FitDiagnostics
from volbench.results import build_config, config_hash
from volbench.splitter import RollingOriginSplitter


def _config(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model_name": "m",
        "model_spec": {"k": 1},
        "data_spec": {"asset": "A"},
        "splitter": RollingOriginSplitter(window=10, horizon=1, step=1, refit_every=1),
        "seed": 7,
        "version": "test",
    }
    kwargs.update(overrides)
    return build_config(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# resolving the pin
# --------------------------------------------------------------------------


class TestThreadPin:
    def test_openblas_wins_over_omp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenBLAS's own precedence, mirrored — the resolved value has to be
        the one the library will actually use, not the one we would prefer."""
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
        monkeypatch.setenv("OMP_NUM_THREADS", "9")
        assert thread_pin() == 3

    def test_omp_is_used_when_openblas_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        assert thread_pin() == 4

    def test_unpinned_falls_back_to_the_cpu_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in THREAD_PIN_VARS:
            monkeypatch.delenv(var, raising=False)
        assert thread_pin() == (os.cpu_count() or 1)
        assert is_pinned() is False

    @pytest.mark.parametrize("bad", ["", "   ", "0", "-2", "auto", "1.5"])
    def test_a_malformed_pin_is_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch, bad: str
    ) -> None:
        """What the BLAS itself does with it. Inventing a number here would
        record a pin that is not in force — a config that describes a machine
        the run was not on."""
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", bad)
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        assert thread_pin() == (os.cpu_count() or 1)
        assert is_pinned() is False

    def test_the_pin_is_reported_as_explicit_when_it_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", str(PINNED_THREADS))
        assert is_pinned() is True
        assert environment_report()["thread_pin_explicit"] is True


# --------------------------------------------------------------------------
# the pin is in the hash
# --------------------------------------------------------------------------


class TestThreadCountIsHashed:
    def test_two_thread_counts_cannot_share_a_config_hash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect this decision closes, stated as an assertion. Before
        D-032 these two were the same string."""
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
        one = config_hash(_config())
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
        eight = config_hash(_config())
        assert one != eight

    def test_the_same_pin_reproduces_the_same_hash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
        first = config_hash(_config())
        monkeypatch.setenv("OMP_NUM_THREADS", "1")  # agrees; changes nothing
        assert config_hash(_config()) == first

    def test_the_pin_reaches_the_hash_through_the_environment_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
        assert _config()["environment"] == {"blas_threads": 2}
        assert environment_spec() == {"blas_threads": 2}

    def test_a_caller_can_pin_the_environment_for_a_test(self) -> None:
        """So a hash can be pinned without pinning the host that computes it."""
        frozen = _config(environment={"blas_threads": 1})
        assert frozen["environment"] == {"blas_threads": 1}

    def test_the_environment_block_is_recorded_even_when_it_looks_inert(self) -> None:
        """The opposite of ``protocol``'s rule, and deliberately so: a setting
        that changes a number always binds, so there is no configuration under
        which leaving it out would be safe."""
        assert "environment" in _config()


# --------------------------------------------------------------------------
# what a run records about its machine
# --------------------------------------------------------------------------


class TestEnvironmentReport:
    def test_it_names_the_blas_build_and_version(self) -> None:
        """The manifest has to answer "which BLAS?", not only "how many
        threads?" — two builds at one thread count can still disagree."""
        info = blas_info()
        assert "name" in info and "version" in info, info

    def test_it_carries_the_pins_and_the_kernel_signature(self) -> None:
        report = environment_report()
        assert set(report["env"]) == {KERNEL_PIN_VAR, *THREAD_PIN_VARS}
        assert isinstance(report["kernel_signature"], str)
        assert report["blas_threads"] == thread_pin()

    def test_it_is_json_serializable(self) -> None:
        """It is written into a manifest, which is JSON."""
        json.loads(json.dumps(environment_report(), default=str))

    def test_the_worker_environment_carries_both_pins(self) -> None:
        """A pool that propagated only the kernel pin would let its workers run
        at a different thread count than the parent that hashed the config."""
        assert set(determinism_env()) == {KERNEL_PIN_VAR, *THREAD_PIN_VARS}


class TestWorkersAreCheckedAgainstTheParent:
    def test_a_worker_at_a_different_thread_count_refuses_the_work(self) -> None:
        """Belt to the propagation's brace. If something outside
        :mod:`volbench.execute` overrode a worker's pin, its numbers would be
        filed under a hash describing the parent's machine — so it must refuse
        loudly rather than write them."""
        from volbench.execute import _apply, _Payload

        def double(x: int) -> int:
            return 2 * x

        honest = _Payload(fn=double, item=21, kernel=None, threads=thread_pin())
        assert _apply(honest) == 42

        impostor = _Payload(fn=double, item=21, kernel=None, threads=thread_pin() + 1)
        with pytest.raises(RuntimeError, match="BLAS thread count"):
            _apply(impostor)


# --------------------------------------------------------------------------
# the disjunction, end to end, at two real thread counts
# --------------------------------------------------------------------------

_CELL = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    from volbench.evaluate import run_backtest
    from volbench.models.garch import GARCH
    from volbench.splitter import RollingOriginSplitter

    def factory():
        return GARCH(o=0, dist="studentst")

    rng = np.random.default_rng(11)
    n = 320
    sigma2 = np.empty(n); eps = np.empty(n)
    sigma2[0] = 1e-4
    for t in range(n):
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        if t + 1 < n:
            sigma2[t + 1] = 1e-6 + 0.08 * eps[t] ** 2 + 0.90 * sigma2[t]
    proxy = np.maximum(eps ** 2, 1e-12)

    frame = run_backtest(
        factory,
        eps,
        proxy,
        RollingOriginSplitter(window=280, horizon=1, step=1, refit_every=1),
        7,
        asset="X",
        proxy_name="sq",
        levels=(0.05,),
    )
    print(json.dumps({
        "config_hash": frame.attrs["config_hash"],
        "forecast_var": [float(v) for v in frame["forecast_var"]],
        "fit_status": [str(v) for v in frame["fit_status"]],
    }))
    """
)


def _run_cell_at(threads: int, tmp_path: Path) -> dict[str, object]:
    """Score one GARCH-t cell in a fresh process pinned to ``threads`` threads.

    A subprocess is not incidental: OpenBLAS reads its thread count when the
    library loads, so an in-process ``monkeypatch`` would change the recorded
    pin without changing a single arithmetic operation — that is, it would make
    this test pass while testing nothing.
    """
    script = tmp_path / f"cell_{threads}.py"
    script.write_text(_CELL, encoding="utf-8")
    env = dict(os.environ)
    for var in THREAD_PIN_VARS:
        env[var] = str(threads)
    out = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    parsed: dict[str, object] = json.loads(out.stdout.strip().splitlines()[-1])
    return parsed


@pytest.mark.slow_determinism
class TestTwoThreadCountsAreNeverOneAnswer:
    """The gate. Two thread counts must give identical numbers **or** different
    hashes — never the pre-D-032 state of one hash over two answers.

    Run out of process at two real thread counts, so the arithmetic differs if
    it is going to. On a single-core host both runs may genuinely agree; the
    assertion is a disjunction precisely so that it stays true and meaningful
    there rather than becoming a flake.
    """

    def test_same_hash_implies_same_numbers(self, tmp_path: Path) -> None:
        one = _run_cell_at(PINNED_THREADS, tmp_path)
        many = _run_cell_at(4, tmp_path)

        identical = np.array_equal(
            np.asarray(one["forecast_var"], dtype=float),
            np.asarray(many["forecast_var"], dtype=float),
        )
        assert identical or one["config_hash"] != many["config_hash"], (
            "a 1-thread run and a 4-thread run produced different forecasts under "
            "one config_hash — the store would serve either fragment for the other "
            "(D-032)"
        )

    def test_the_hashes_do_differ_at_different_pins(self, tmp_path: Path) -> None:
        """Stronger than the disjunction, and the reason it is satisfiable: the
        thread count is *in* the hash, so the two runs are two cells whatever
        their numbers do."""
        assert _run_cell_at(1, tmp_path)["config_hash"] != (
            _run_cell_at(4, tmp_path)["config_hash"]
        )

    def test_the_bounded_nu_keeps_the_two_runs_close(self, tmp_path: Path) -> None:
        """The source fix, not the pin (D-032 item 3). Before the ``nu`` bound
        the same comparison moved by 5.5e-1 relative on the toy fixture; a
        Student-t whose degrees of freedom cannot wander into the flat part of
        the likelihood no longer hands SLSQP a direction to be pushed along.

        The tolerance is loose on purpose — this pins "no different local
        optimum", not a particular ulp count, and a different BLAS build is
        entitled to its own last bits."""
        one = np.asarray(_run_cell_at(1, tmp_path)["forecast_var"], dtype=float)
        many = np.asarray(_run_cell_at(4, tmp_path)["forecast_var"], dtype=float)
        rel = np.abs(one - many) / np.abs(one)
        assert rel.max() < 1e-3, f"max relative difference {rel.max():.3e}"


# --------------------------------------------------------------------------
# a fit that fell back is no longer silent
# --------------------------------------------------------------------------


class TestFitStatusVocabulary:
    def test_a_clean_fit_says_ok(self) -> None:
        assert FitDiagnostics(converged=True).status() == "ok"

    def test_a_fallback_names_the_estimator_that_ran(self) -> None:
        status = FitDiagnostics(converged=False, fallback="ewma", detail="flag=9").status()
        assert status == "fallback=ewma|flag=9"
        assert FitDiagnostics.is_fallback(status)
        assert FitDiagnostics.is_nonconverged(status)

    def test_non_convergence_without_a_fallback_is_expressible(self) -> None:
        status = FitDiagnostics(converged=False).status()
        assert status == "nonconverged"
        assert not FitDiagnostics.is_fallback(status)
        assert FitDiagnostics.is_nonconverged(status)

    def test_the_empty_string_is_reserved_for_models_that_report_nothing(self) -> None:
        """So an empty ``fit_status`` can never be misread as a clean fit — the
        distinction between "did not fall back" and "did not say"."""
        for diagnostics in (
            FitDiagnostics(converged=True),
            FitDiagnostics(converged=False),
            FitDiagnostics(converged=False, fallback="ewma"),
        ):
            assert diagnostics.status() != ""
        assert not FitDiagnostics.is_nonconverged("")
        assert not FitDiagnostics.is_fallback("")

    def test_detail_is_collapsed_to_one_line(self) -> None:
        """It lands in a results column that people read in a terminal."""
        status = FitDiagnostics(converged=True, detail="a\n  b\tc").status()
        assert status == "ok|a b c"
