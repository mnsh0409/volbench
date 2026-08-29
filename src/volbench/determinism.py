"""What the *machine* contributes to a number, made explicit (D-026, D-032).

Everything else in volbench treats a result as a function of the model, the
data, the protocol and the seed. Two machine-level settings break that, and
both were found by measurement rather than reasoning:

**The numpy SIMD kernel family (D-026).** numpy's AVX-512-only float64
``log``/``exp`` kernels differ from the x86-v3 ones in the last ulp for some
inputs. That moves the *content digest* of a computed proxy and with it every
config hash built on it, so the two machines disagree by **missing** each
other's cache — a wasted grid, never a wrong answer.
:func:`kernel_signature` is what a worker checks itself against.

**The BLAS thread count (D-032).** Threaded OpenBLAS reorders a reduction by
an ulp; ``arch``'s SLSQP turns that into a different local optimum of the
GARCH likelihood. Unlike the kernel family this moves **no** digest, so before
D-032 a 32-thread fragment and a 1-thread fragment of one cell shared one
config hash and the store was free to serve either for the other — two answers
under one name. :func:`thread_pin` is therefore *hashed*
(:func:`environment_spec`), which converts that silent substitution into the
same honest cache miss D-026 already produces.

The pin itself — ``OMP_NUM_THREADS=1``, ``OPENBLAS_NUM_THREADS=1`` — is set in
the Makefile, in CI and in every worker (:mod:`volbench.execute`), so the
sanctioned path records ``1`` on every machine and cross-machine cache sharing
(D-011) survives. It is the *unpinned* path that stops sharing, which is the
point: an unpinned run's numbers are a property of its core count.

This module deliberately imports nothing from :mod:`volbench`, so every layer
that needs it — results, execute, runner — can.
"""

from __future__ import annotations

import hashlib
import os
import platform
from typing import Any, Final

__all__ = [
    "KERNEL_PIN_VAR",
    "PINNED_THREADS",
    "THREAD_PIN_VARS",
    "blas_info",
    "determinism_env",
    "environment_report",
    "environment_spec",
    "interpreter_info",
    "kernel_signature",
    "thread_pin",
]

#: The environment variable D-026 pins the numpy SIMD kernel family with.
KERNEL_PIN_VAR: Final = "NPY_DISABLE_CPU_FEATURES"

#: The variables D-032 pins the BLAS thread count with, in the precedence
#: OpenBLAS itself applies: its own variable wins over the generic OpenMP one.
#: Both are pinned rather than only ``OPENBLAS_NUM_THREADS`` because the
#: OpenMP one also governs LightGBM's thread pool, and a grid whose workers
#: each spawn 32 OpenMP threads is oversubscribed long before it is wrong.
THREAD_PIN_VARS: Final = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")

#: What the pin sets them to. One thread: the only value every machine can
#: honour, and the one under which serial and pooled runs cannot diverge.
PINNED_THREADS: Final = 1


# --------------------------------------------------------------------------
# numpy kernel family (D-026)
# --------------------------------------------------------------------------


def kernel_signature() -> str:
    """Digest of the numpy SIMD dispatch targets *enabled in this process*.

    Two processes sharing this string compute ``log``/``exp`` with the same
    kernels and therefore agree bit for bit; two that do not can disagree in
    the last ulp, which moves every content digest downstream (D-026).

    ``numpy._core._multiarray_umath`` is private, so an unreadable or
    restructured build degrades to a signature over the pin variable itself
    rather than raising: this is a guard, and a guard that cannot run must not
    take the run down with it.
    """
    try:
        from numpy._core._multiarray_umath import (  # type: ignore[import-not-found]
            __cpu_baseline__,
            __cpu_dispatch__,
            __cpu_features__,
        )
    except Exception:  # pragma: no cover - numpy internals moved or unreadable
        return "unknown:" + os.environ.get(KERNEL_PIN_VAR, "")
    enabled = [target for target in sorted(__cpu_dispatch__) if __cpu_features__.get(target)]
    canonical = "baseline=" + ",".join(sorted(__cpu_baseline__)) + ";dispatch=" + ",".join(enabled)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]


# --------------------------------------------------------------------------
# BLAS thread count (D-032)
# --------------------------------------------------------------------------


def _positive_int(raw: str | None) -> int | None:
    """``raw`` as a positive int, or ``None`` if it is not one.

    A malformed pin (``OMP_NUM_THREADS=""``, ``=auto``, ``=0``) is treated as
    *unset*, which is what the BLAS itself does with it. Guessing a number
    here would record a pin that is not in force.
    """
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError):
        return None
    return value if value > 0 else None


def thread_pin() -> int:
    """The BLAS thread count this process will actually run at.

    Resolution order is OpenBLAS's own: ``OPENBLAS_NUM_THREADS``, then
    ``OMP_NUM_THREADS``, then — nothing being pinned — the machine's CPU count,
    which is what an unpinned OpenBLAS defaults to.

    That last branch is the one that matters for the hash. It is deliberately
    read from :func:`os.cpu_count` and **not** from a runtime introspection of
    the loaded BLAS: the hashed value must be a function of the environment
    and the machine alone, never of whether an optional introspection package
    happens to be installed. Under the D-032 pin the branch is never taken, so
    the exactness of the unpinned figure only decides *which* cache miss an
    unpinned run gets, never whether it gets one.
    """
    for var in THREAD_PIN_VARS:
        value = _positive_int(os.environ.get(var))
        if value is not None:
            return value
    return os.cpu_count() or 1


def is_pinned() -> bool:
    """Whether an explicit thread pin is in force (as opposed to inferred)."""
    return any(_positive_int(os.environ.get(var)) is not None for var in THREAD_PIN_VARS)


def determinism_env() -> dict[str, str | None]:
    """The pins a worker must reapply verbatim to agree with this process.

    ``None`` means *unset*, and a worker sets it unset — inventing a value the
    parent did not have would make the pool disagree with a serial run in the
    same shell, which is precisely the failure this exists to prevent.
    """
    keys = (KERNEL_PIN_VAR, *THREAD_PIN_VARS)
    return {key: os.environ.get(key) for key in keys}


# --------------------------------------------------------------------------
# what the run records about its machine
# --------------------------------------------------------------------------


def blas_info() -> dict[str, Any]:
    """The BLAS numpy is built against: name, version, and how it is threaded.

    Read from ``numpy.__config__``, which is generated at build time and is
    the only public place this is available. Degrades to ``{}`` rather than
    raising, for the same reason :func:`kernel_signature` does.
    """
    try:
        import numpy as np

        config: Any = np.__config__.show(mode="dicts")
        blas: dict[str, Any] = dict(config["Build Dependencies"]["blas"])
    except Exception:  # pragma: no cover - numpy internals moved or unreadable
        return {}
    keep = ("name", "version", "openblas configuration", "detection method")
    return {key: blas[key] for key in keep if key in blas}


def _observed_thread_pools() -> list[dict[str, Any]] | None:
    """The native thread pools actually loaded, if ``threadpoolctl`` is present.

    Ground truth rather than the resolved pin, and reported *only* in the
    manifest — never hashed — because it is available exactly when an optional
    package is installed, and a hash may not depend on that.
    """
    try:
        import threadpoolctl
    except Exception:
        return None
    try:
        pools = threadpoolctl.threadpool_info()
    except Exception:  # pragma: no cover - introspection failed on this build
        return None
    keep = ("internal_api", "prefix", "version", "num_threads", "threading_layer", "architecture")
    return [{key: pool[key] for key in keep if key in pool} for pool in pools]


def interpreter_info() -> dict[str, Any]:
    """Which Python ran this, for the **manifest** — reported, never hashed.

    A run that records no interpreter forces anyone checking it to establish
    one indirectly (docs/P3_ANALYSIS_VALIDITY.md had to do it three ways), and
    an indirect answer is not evidence. It stays out of
    :func:`environment_spec` deliberately: the version is not known to move a
    number here — the package supports 3.11-3.13 and CI runs all three — so
    hashing it would split the cache three ways for a claim nothing has
    measured. It is recorded so a disagreement between two runs can be
    attributed rather than guessed at.

    ``sys.executable`` used to be recorded here as well, to say which *venv*.
    It is not any more. It is an absolute path under a home directory, so it
    names the person who ran the study, and IJF review is double-blind: the
    reproducibility package is part of what a reviewer sees
    (``tests/test_identity_leakage.py``). Nothing is lost that reproduces a
    run — the *version* is what a reader needs, and the venv path is meaningful
    only on the one machine that already has it.
    """
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def environment_spec() -> dict[str, Any]:
    """The machine settings that enter the **config hash** (D-032).

    One key today. It is deliberately minimal: everything in here splits the
    cache, so a setting earns its place only by having been *measured* to move
    a number. ``blas_threads`` was — see docs/P3_DETERMINISM.md §2. The numpy
    kernel family is not here because it already reaches the hash through the
    content digests of every computed proxy (D-026).
    """
    return {"blas_threads": thread_pin()}


def environment_report() -> dict[str, Any]:
    """The fuller machine record for a **run manifest** — reported, not hashed.

    The manifest may say more than the hash, and should: a reader diagnosing
    two runs that disagree needs the BLAS build, the interpreter and the pins
    as they stood, not only the one integer that split the cache.
    """
    report: dict[str, Any] = {
        "blas_threads": thread_pin(),
        "thread_pin_explicit": is_pinned(),
        "kernel_signature": kernel_signature(),
        "interpreter": interpreter_info(),
        "cpu_count": os.cpu_count(),
        "env": {key: value for key, value in sorted(determinism_env().items())},
        "blas": blas_info(),
    }
    pools = _observed_thread_pools()
    if pools is not None:
        report["observed_thread_pools"] = pools
    return report
