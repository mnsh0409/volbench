"""Opt-in gates for the tests that need model weights, a GPU, or the network.

Two markers, both registered in ``pyproject.toml``:

``@pytest.mark.tsfm``
    Loads real foundation-model weights (Chronos / TimesFM / Moirai): needs
    the ``tsfm`` extra installed, a Hugging Face download on first use, and a
    GPU to be quick. Skipped unless ``VOLBENCH_RUN_TSFM=1`` is set.
``@pytest.mark.timegpt``
    Calls Nixtla's TimeGPT API: skipped unless ``NIXTLA_API_KEY`` is set (the
    adapter reads the same variable — never a key in code or fixtures).
``@pytest.mark.gpu``
    Trains the PatchTST baseline on a CUDA device at its real size: skipped
    unless ``VOLBENCH_RUN_GPU=1`` (a 2-epoch CPU smoke test covers it in CI).
``@pytest.mark.pinned_identity``
    Asserts a committed ``config_hash``. Since D-032 the BLAS thread count is
    part of every hash, so these identities are only *defined* under the pin
    the Makefile and CI export (``OMP_NUM_THREADS=1``,
    ``OPENBLAS_NUM_THREADS=1``). On an unpinned shell they are skipped with a
    message naming the pin, rather than failing: the machine has not
    contradicted the committed identity, it has computed a different one.

All are additionally skipped whenever ``CI`` is set (GitHub Actions sets
``CI=true``), so a stray key or opt-in flag in a CI secret can never make the
suite download weights or call the network there. The default ``uv run
pytest`` therefore stays green with no GPU and no network; each adapter keeps
a mocked-backend test outside these markers so its contract is covered in CI.
"""

from __future__ import annotations

import os

import pytest

from volbench.determinism import PINNED_THREADS, THREAD_PIN_VARS, thread_pin

TSFM_OPT_IN = "VOLBENCH_RUN_TSFM"
GPU_OPT_IN = "VOLBENCH_RUN_GPU"
TIMEGPT_KEY = "NIXTLA_API_KEY"


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {"", "0", "false", "no"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    in_ci = _truthy(os.environ.get("CI"))
    run_tsfm = _truthy(os.environ.get(TSFM_OPT_IN)) and not in_ci
    run_timegpt = bool(os.environ.get(TIMEGPT_KEY)) and not in_ci
    run_gpu = _truthy(os.environ.get(GPU_OPT_IN)) and not in_ci
    skip_tsfm = pytest.mark.skip(
        reason=f"needs model weights/GPU: set {TSFM_OPT_IN}=1 (never honoured under CI)"
    )
    skip_timegpt = pytest.mark.skip(
        reason=f"needs a TimeGPT key: set {TIMEGPT_KEY} (never honoured under CI)"
    )
    skip_gpu = pytest.mark.skip(
        reason=f"trains on a GPU: set {GPU_OPT_IN}=1 (never honoured under CI)"
    )
    threads = thread_pin()
    skip_unpinned = pytest.mark.skip(
        reason=(
            f"committed identities are defined at {PINNED_THREADS} BLAS thread "
            f"(D-032); this shell resolves to {threads}. Run `make check`, or set "
            + " ".join(f"{var}={PINNED_THREADS}" for var in THREAD_PIN_VARS)
        )
    )
    for item in items:
        if not run_tsfm and item.get_closest_marker("tsfm") is not None:
            item.add_marker(skip_tsfm)
        if not run_timegpt and item.get_closest_marker("timegpt") is not None:
            item.add_marker(skip_timegpt)
        if not run_gpu and item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)
        if threads != PINNED_THREADS and item.get_closest_marker("pinned_identity"):
            item.add_marker(skip_unpinned)
