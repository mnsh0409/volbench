"""GARCH(1,1) / GJR-GARCH(1,1,1) via the `arch` package.

Contract: `fit(train)` takes a 1-D array of returns. Fit with `mean="Zero"`
(pure variance dynamics, consistent with the other baselines here).

Units: `arch_model` is fit with `rescale=True` unconditionally rather than
relying on its default auto-rescale (`rescale=None`, which rescales *and
warns* whenever it decides the data is poorly scaled) so the optimizer
always sees a well-conditioned series regardless of the caller's return
units. The catch: `res.forecast(...).variance` is reported in the
*rescaled* series' units, not the caller's — dividing by `res.scale ** 2`
before returning is required, or this would silently violate the
daily-units-only rule (CLAUDE.md rule 2).

Student-t innovations: the predictive distribution is the parametric
`StudentT` (location 0, `nu` from the fit, scale derived from the conditional
variance via `StudentT.from_variance`, so that its `variance()` is exactly
`arch`'s conditional variance). Until m2/evaluator-hardening this was a
199-point quantile grid over tau in [0.005, 0.995]; the evaluator's moments of
that grid truncated the tails, and a *perfectly specified* forecast could not
score below a QLIKE floor of 0.0407 at nu=3 (docs/M1_REPORT.md §4.2). The
parametric object has closed-form moments and CRPS and needs no RNG, so the
same fitted model still scores bit-identically across runs (CLAUDE.md rule 3).

Non-convergence: if the optimizer fails to converge (`convergence_flag !=
0`, including a degenerate fitted `nu <= 2` for Student-t, where the
predictive variance would be undefined) or `fit()` raises outright, this
falls back to `EWMA(lambda_=fallback_lambda)` on the same training window
and records `fallback=True` on the fitted object, per the HARD RULES for
this stream: an origin must get a usable forecast, never a raised
exception that drops it from the grid. A fit that fell back is no longer
silent: `fit_diagnostics()` reports it, the evaluator records it per row in
`fit_status`, and the runner counts it per cell (D-032).

What a fallback retains, and why it retains it: the optimizer's terminal
state — the parameter vector it stopped at, its final log-likelihood, its
exit flag and message, and the box bounds it was run against — is kept on
:class:`TerminalFit` for *every* scheduled fit, converged or not. Until
this was added a fit that fell back discarded `result` outright, so the
only surviving evidence about it was the exit flag in `fit_status`, and
the question "which parameter was at which bound when SLSQP stopped?" was
not answerable from any artifact — only guessable. `terminal` is never
hashed, never in `spec()`, and never read by `predict` or `update`; it
changes no number and moves no config hash. Like `detail` it describes the
*scheduled* fit, so `update` carries it forward unchanged.

Identification, and why it is a reproducibility matter (D-032): `nu` is
bounded to `NU_BOUNDS = (2.1, 50)` and SLSQP is run at `FIT_TOL = 1e-10`.
Both are hyperparameters in `spec()`, so they are in the config hash. The
reason is measured rather than stylistic. `arch`'s own upper bound on `nu` is
500, and a 500-observation window carries no information about tail thickness
up there — at nu = 50 a Student-t already matches a Gaussian to past float
precision in every quantile this project scores. The likelihood is therefore
flat along `nu`, SLSQP wanders in that flat direction, and the wandering
couples into (omega, alpha, beta) hard enough that a *last-ulp* difference in
a threaded BLAS reduction moves the fit into a different local optimum: on the
toy fixture, `forecast_var` moved by 5.5e-1 relative on two of 200 origins,
where the normal-innovations model on the same data moved by 9.2e-5.
Bounding `nu` cuts that to 2.3e-4 and the tighter tolerance to 2.9e-6
(docs/P3_DETERMINISM.md §4). A multi-start over several starting values with
best-likelihood selection was measured too and is deliberately NOT used: it
made the thread sensitivity worse, because taking a max over twelve
independently thread-sensitive optimizations is more variable than one, and it
found a better optimum on 1 of 200 origins for 12x the fits.

Re-conditioning between refits (docs/M1_REPORT.md §4.3): `update(train)`
re-filters the conditional variance over the current window at the
parameters estimated at the last scheduled refit, through `arch`'s
`ARCHModel.fix` — the same specification rebuilt on the new window and
evaluated at fixed parameters, with no optimizer involved. The window is
pre-multiplied by the fit's `scale` and the fixed model is built with
`rescale=False`, so the parameters keep the units they were estimated in
and `predict` undoes the scale exactly as before. On the fit's own window
`fix` reproduces the fitted forecast and every in-sample conditional
variance to the bit, which is also the proof that the scale handling is
right (tests/test_models_update.py).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import StudentsT
from arch.univariate.base import ARCHModelFixedResult, ARCHModelResult
from numpy.typing import NDArray

from volbench.dist import Distribution, Normal, StudentT
from volbench.models.base import FitDiagnostics
from volbench.models.ewma import EWMA, FittedEWMA

__all__ = ["FIT_TOL", "GARCH", "NU_BOUNDS", "FittedGARCH", "TerminalFit", "gjr_garch"]

logger = logging.getLogger(__name__)

_Dist = Literal["normal", "studentst"]
_MIN_TRAIN = 20
_MIN_NU = 2.02  # Student-t variance nu/(nu-2) blows up as nu -> 2

#: Bounds on the Student-t degrees of freedom (D-032). ``arch``'s own are
#: ``(2.05, 500)``, and the upper end of that range is not a range the data
#: can speak to: at nu = 500 a Student-t is a Gaussian to well past float
#: precision in every quantile this project scores. Estimating into it makes
#: the likelihood flat along nu, and a flat direction is what let a
#: last-ulp BLAS difference move SLSQP to a different local optimum — 5.5e-1
#: relative on a toy `garch11_t` forecast (docs/P3_DETERMINISM.md §2).
#:
#: 50 is where the identification argument bites, not where the numbers
#: stopped moving: it is chosen because nu above it is not estimable from a
#: 500-observation window, and the measurement was run afterwards. 2.1 keeps
#: the predictive variance nu/(nu-2) finite and well away from its pole.
#: Both ends are in ``spec()``, so a run that moves them is a different cell.
NU_BOUNDS: Final[tuple[float, float]] = (2.1, 50.0)

#: SLSQP's convergence tolerance. ``scipy``'s default is 1e-6; this is
#: *tighter*, never looser — a tolerance widened until a fit "passes" would
#: hide the non-convergence this module falls back on rather than fix it.
#: Worth about two further orders of magnitude of thread-stability on top of
#: the nu bound, and no measured non-convergence (docs/P3_DETERMINISM.md §4).
FIT_TOL: Final = 1e-10


def _spec(o: int, dist: _Dist, fallback_lambda: float, fit_tol: float) -> dict[str, Any]:
    """The part of the spec both the model and its fit agree on."""
    return {
        "model": "gjr_garch" if o else "garch",
        "p": 1,
        "o": o,
        "q": 1,
        "dist": dist,
        "fallback_lambda": fallback_lambda,
        "fit_tol": fit_tol,
    }


class _BoundedStudentsT(StudentsT):
    """``arch``'s Student-t with the degrees of freedom bounded to :data:`NU_BOUNDS`.

    Only the optimizer's box constraint changes; the density, and therefore
    every likelihood and every forecast evaluated *at* a parameter vector, is
    ``arch``'s own. Module-level (not a closure) so a fitted result stays
    picklable across the process executor's boundary.
    """

    def __init__(self, bounds: tuple[float, float] = NU_BOUNDS) -> None:
        super().__init__()
        self._nu_bounds = (float(bounds[0]), float(bounds[1]))

    def bounds(self, resids: NDArray[np.float64] | pd.Series) -> list[tuple[float, float]]:
        return [self._nu_bounds]


@dataclass(frozen=True)
class TerminalFit:
    """Where SLSQP stopped on one scheduled fit, converged or not.

    Every field is read straight off ``arch``'s own result and
    ``scipy``'s ``OptimizeResult`` underneath it; nothing here is derived,
    re-estimated or rounded. Held as plain tuples and floats so the whole
    record pickles across :mod:`volbench.execute`'s process boundary as
    cheaply as the flag it replaces.

    Attributes
    ----------
    params:
        ``(name, value)`` in ``arch``'s own parameter order, on the
        **rescaled** series the optimizer saw (see the module docstring):
        ``omega`` is in units of ``scale ** 2``, and ``alpha``, ``gamma``,
        ``beta`` and ``nu`` are scale-free. :attr:`omega_on_the_return_scale`
        undoes it.
    loglikelihood:
        The log-likelihood at :attr:`params`, of the rescaled series.
    convergence_flag:
        ``arch``'s ``convergence_flag`` — ``scipy``'s SLSQP ``status``. ``0``
        is convergence; ``8`` is "positive directional derivative for line
        search".
    message:
        SLSQP's own description of :attr:`convergence_flag`.
    iterations, function_evals:
        SLSQP's ``nit`` and ``nfev``, which separate a fit that ran out of
        iterations from one that stopped because it could not descend.
    scale:
        The factor ``arch``'s ``rescale=True`` multiplied the returns by.
    bounds:
        ``(name, low, high)`` — the box the optimizer was run inside, in the
        same order as :attr:`params`, so "at a bound" is decidable from this
        record alone rather than from a re-derivation of ``arch``'s formulas.
    """

    params: tuple[tuple[str, float], ...]
    loglikelihood: float
    convergence_flag: int
    message: str
    iterations: int
    function_evals: int
    scale: float
    bounds: tuple[tuple[str, float, float], ...]

    def __post_init__(self) -> None:
        if len(self.params) != len(self.bounds):
            raise ValueError(
                f"{len(self.params)} parameters but {len(self.bounds)} bounds; a bound must "
                "be recorded for every parameter or 'at a bound' cannot be decided"
            )
        for (name, _), (bound_name, _, _) in zip(self.params, self.bounds, strict=True):
            if name != bound_name:
                raise ValueError(f"parameter {name!r} is paired with the bound of {bound_name!r}")

    @property
    def omega_on_the_return_scale(self) -> float:
        """:attr:`params`' ``omega``, divided back out of ``scale ** 2``."""
        return dict(self.params)["omega"] / (self.scale**2)

    def slack(self) -> dict[str, tuple[float, float]]:
        """Per parameter, ``(value - low, high - value)``: distance to each bound.

        Both entries are non-negative for a parameter strictly inside its box.
        A negative one means the optimizer stopped a hair outside the box,
        which SLSQP allows within its own constraint tolerance.
        """
        return {
            name: (value - low, high - value)
            for (name, value), (_, low, high) in zip(self.params, self.bounds, strict=True)
        }

    def as_dict(self) -> dict[str, Any]:
        """A flat, JSON-safe record — one row of a forensics table."""
        record: dict[str, Any] = {
            "loglikelihood": self.loglikelihood,
            "convergence_flag": self.convergence_flag,
            "message": self.message,
            "iterations": self.iterations,
            "function_evals": self.function_evals,
            "scale": self.scale,
        }
        for name, value in self.params:
            record[name] = value
        for name, low, high in self.bounds:
            record[f"{name}_low"] = low
            record[f"{name}_high"] = high
        return record


def _terminal_fit(
    model: Any, result: ARCHModelResult, resids: NDArray[np.float64]
) -> TerminalFit:
    """The optimizer's terminal state, read off ``arch``'s result.

    ``resids`` is the **rescaled** series the optimizer saw, which for a
    zero-mean model is the series itself: that is what ``arch`` hands its own
    ``bounds()`` methods, so passing it here reproduces the box SLSQP was run
    inside rather than approximating it. ``arch`` computes the *distribution's*
    bounds from standardized residuals instead; both distributions this module
    uses ignore the argument entirely — ``Normal`` has no parameters and
    :class:`_BoundedStudentsT` returns :data:`NU_BOUNDS` unconditionally — so
    the two agree here by construction, and
    ``tests/test_models_garch.py::TestTerminalFit`` pins that.
    """
    optimizer = result.optimization_result
    bounds = [*model.bounds(), *model.volatility.bounds(resids), *model.distribution.bounds(resids)]
    names = [str(name) for name in result.params.index]
    return TerminalFit(
        params=tuple(zip(names, (float(v) for v in result.params), strict=True)),
        loglikelihood=float(result.loglikelihood),
        convergence_flag=int(result.convergence_flag),
        message=str(getattr(optimizer, "message", "")),
        iterations=int(getattr(optimizer, "nit", -1)),
        function_evals=int(getattr(optimizer, "nfev", -1)),
        scale=float(result.scale),
        bounds=tuple(
            (name, float(low), float(high)) for name, (low, high) in zip(names, bounds, strict=True)
        ),
    )


@dataclass(frozen=True)
class FittedGARCH:
    name: str
    o: int
    dist: _Dist
    fallback_lambda: float
    fallback: bool
    #: The scheduled fit (an ``ARCHModelResult``) or, after ``update``, the
    #: same parameters fixed on a newer window (an ``ARCHModelFixedResult``,
    #: which the fitted result subclasses). ``None`` on a fallback fit.
    result: ARCHModelFixedResult | None
    scale: float
    fallback_fit: FittedEWMA | None
    nu_bounds: tuple[float, float] = NU_BOUNDS
    fit_tol: float = FIT_TOL
    #: Free-form evidence about the scheduled fit, surfaced through
    #: :meth:`fit_diagnostics` into the results' ``fit_status`` column (D-032).
    #: Never hashed and never read by any code path that produces a number.
    detail: str = ""
    #: Where the optimizer stopped on the scheduled fit — retained whether or
    #: not it converged, so a fallback can be diagnosed after the fact instead
    #: of guessed at (see the module docstring). ``None`` only when ``fit``
    #: itself raised, since there is then no optimizer result to read. Never
    #: hashed, never in :meth:`spec`, and never read by :meth:`predict` or
    #: :meth:`update`.
    terminal: TerminalFit | None = None

    def fit_diagnostics(self) -> FitDiagnostics:
        """How the scheduled fit went (D-032).

        A property of the *fit*, so :meth:`update` carries it forward unchanged:
        re-conditioning runs no optimizer, and a window re-filtered at the
        parameters of a fit that fell back is still a fallback forecast.
        """
        return FitDiagnostics(
            converged=not self.fallback,
            fallback="ewma" if self.fallback else "",
            detail=self.detail,
        )

    def spec(self) -> dict[str, Any]:
        spec = _spec(self.o, self.dist, self.fallback_lambda, self.fit_tol)
        if self.dist == "studentst":
            spec["nu_bounds"] = [float(self.nu_bounds[0]), float(self.nu_bounds[1])]
        return spec

    def predict(self, h: int) -> Distribution:
        if h < 1:
            raise ValueError("h must be >= 1")
        if self.fallback:
            assert self.fallback_fit is not None
            return self.fallback_fit.predict(h)
        assert self.result is not None

        forecast = self.result.forecast(horizon=h, reindex=False)
        sigma2 = float(forecast.variance.values[-1, h - 1]) / (self.scale**2)
        if self.dist == "normal":
            return Normal(mu=0.0, sigma=math.sqrt(sigma2))
        nu = float(self.result.params["nu"])
        return StudentT.from_variance(0.0, sigma2, nu)

    def update(self, train: NDArray[np.float64]) -> FittedGARCH:
        """Re-filter the conditional variance over ``train`` at fixed parameters.

        Nothing is re-estimated: the parameters (including ``nu``) and the
        ``scale`` are those of the last scheduled fit; only the variance path
        — and therefore the forecast — is re-conditioned on the new window.
        A fallback fit re-conditions its EWMA instead. See the module
        docstring for the ``rescale`` handling.
        """
        arr = np.asarray(train, dtype=np.float64)
        if arr.ndim != 1 or arr.size < _MIN_TRAIN:
            raise ValueError(f"train must be a 1-D array with at least {_MIN_TRAIN} returns")
        if self.fallback:
            assert self.fallback_fit is not None
            return dataclasses.replace(self, fallback_fit=self.fallback_fit.update(arr))
        assert self.result is not None
        am = arch_model(
            arr * self.scale,
            mean="Zero",
            vol="GARCH",
            p=1,
            o=self.o,
            q=1,
            dist=self.dist,
            rescale=False,
        )
        if self.dist == "studentst":
            # Cosmetic, deliberately: `fix` evaluates at given parameters and
            # runs no optimizer, so a box constraint cannot reach a number here.
            # Matching the fit's distribution object keeps the two descriptions
            # of one model from drifting apart.
            am.distribution = _BoundedStudentsT(self.nu_bounds)
        return dataclasses.replace(self, result=am.fix(self.result.params))


@dataclass(frozen=True)
class GARCH:
    """GARCH(1,1) (o=0) / GJR-GARCH(1,1,1) (o=1), normal or Student-t innovations."""

    o: int = 0
    dist: _Dist = "normal"
    fallback_lambda: float = 0.94
    #: Box constraint on the Student-t degrees of freedom; ignored for
    #: ``dist="normal"``, which has none. See :data:`NU_BOUNDS`.
    nu_bounds: tuple[float, float] = NU_BOUNDS
    #: SLSQP convergence tolerance. See :data:`FIT_TOL`.
    fit_tol: float = FIT_TOL

    def __post_init__(self) -> None:
        if self.o not in (0, 1):
            raise ValueError("o must be 0 (GARCH) or 1 (GJR-GARCH)")
        if self.dist not in ("normal", "studentst"):
            raise ValueError("dist must be 'normal' or 'studentst'")
        if not 0.0 < self.fallback_lambda < 1.0:
            raise ValueError("fallback_lambda must lie strictly inside (0, 1)")
        low, high = self.nu_bounds
        if not 2.0 < low < high:
            raise ValueError(
                f"nu_bounds must satisfy 2 < low < high, got {self.nu_bounds}; the "
                "predictive variance nu/(nu-2) is undefined at or below nu = 2"
            )
        if not self.fit_tol > 0.0:
            raise ValueError(f"fit_tol must be > 0, got {self.fit_tol}")

    @property
    def name(self) -> str:
        variant = "gjr_garch" if self.o else "garch"
        return f"{variant}(1,1)-{self.dist}"

    def spec(self) -> dict[str, Any]:
        """The hyperparameters, hashed. ``nu_bounds`` appears only where it binds.

        A normal-innovations GARCH has no degrees of freedom to bound, so
        recording the bounds on one would make two identical experiments hash
        differently over a setting neither of them used — the same rule
        ``build_config`` applies to the protocol block.
        """
        spec = _spec(self.o, self.dist, self.fallback_lambda, self.fit_tol)
        if self.dist == "studentst":
            spec["nu_bounds"] = [float(self.nu_bounds[0]), float(self.nu_bounds[1])]
        return spec

    def _model(self, arr: NDArray[np.float64]) -> Any:
        """The ``arch`` model for ``arr``, with this configuration's nu bounds."""
        am = arch_model(
            arr, mean="Zero", vol="GARCH", p=1, o=self.o, q=1, dist=self.dist, rescale=True
        )
        if self.dist == "studentst":
            am.distribution = _BoundedStudentsT(self.nu_bounds)
        return am

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedGARCH:
        arr = np.asarray(train, dtype=np.float64)
        if arr.ndim != 1 or arr.size < _MIN_TRAIN:
            raise ValueError(f"train must be a 1-D array with at least {_MIN_TRAIN} returns")

        result: ARCHModelResult | None = None
        terminal: TerminalFit | None = None
        converged = False
        detail = ""
        try:
            model = self._model(arr)
            result = model.fit(disp="off", show_warning=False, options={"ftol": self.fit_tol})
            terminal = _terminal_fit(model, result, arr * float(result.scale))
            converged = result.convergence_flag == 0
            detail = f"flag={result.convergence_flag}"
            if converged and self.dist == "studentst":
                nu = float(result.params["nu"])
                # Unreachable while nu_bounds[0] > _MIN_NU, which is the point:
                # the bound removes the degenerate-nu failure mode rather than
                # catching it. Kept as the guard for a caller who lowers it.
                converged = nu > _MIN_NU
                at_bound = " nu_at_bound" if nu >= self.nu_bounds[1] * (1 - 1e-9) else ""
                detail = f"flag={result.convergence_flag} nu={nu:.6g}{at_bound}"
        except Exception as exc:
            logger.warning("%s: fit raised, falling back to EWMA", self.name, exc_info=True)
            result = None
            terminal = None
            converged = False
            detail = f"raised {type(exc).__name__}"

        if not converged:
            if result is not None:
                logger.warning(
                    "%s: optimizer did not converge (%s), falling back to EWMA",
                    self.name,
                    detail,
                )
            fallback_fit = EWMA(lambda_=self.fallback_lambda).fit(arr)
            return FittedGARCH(
                name=self.name,
                o=self.o,
                dist=self.dist,
                fallback_lambda=self.fallback_lambda,
                fallback=True,
                result=None,
                scale=1.0,
                fallback_fit=fallback_fit,
                nu_bounds=self.nu_bounds,
                fit_tol=self.fit_tol,
                detail=detail,
                terminal=terminal,
            )

        assert result is not None
        return FittedGARCH(
            name=self.name,
            o=self.o,
            dist=self.dist,
            fallback_lambda=self.fallback_lambda,
            fallback=False,
            result=result,
            scale=float(result.scale),
            fallback_fit=None,
            nu_bounds=self.nu_bounds,
            fit_tol=self.fit_tol,
            detail=detail,
            terminal=terminal,
        )


def gjr_garch(
    dist: _Dist = "normal",
    fallback_lambda: float = 0.94,
    *,
    nu_bounds: tuple[float, float] = NU_BOUNDS,
    fit_tol: float = FIT_TOL,
) -> GARCH:
    return GARCH(
        o=1,
        dist=dist,
        fallback_lambda=fallback_lambda,
        nu_bounds=nu_bounds,
        fit_tol=fit_tol,
    )
