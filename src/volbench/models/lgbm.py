"""Gradient-boosted trees on lagged realized variance, via `lightgbm`.

Contract — SAME INPUT AS ``models/har.py`` AND ``models/sf.py``, NOT THE
RETURN-FED BASELINES: ``fit(train)`` takes a 1-D REALIZED-VARIANCE series in
daily units (CLAUDE.md rule 2). ``run_backtest(..., fit_series=...)`` routes
it, slicing that series with the same ``RollingOriginSplitter`` origins as the
return series.

What is modelled
================

``log RV_{t+1} = f(x_t) + e_t`` with an L2 objective, where ``x_t`` is the
HAR information set widened into something a tree ensemble can use:

    x_t = [ log RV_t, log RV_{t-1}, ..., log RV_{t-21},        (lags 1..22)
            log mean(RV_{t-4 .. t}),                           (HAR weekly)
            log mean(RV_{t-21 .. t}) ]                         (HAR monthly)

24 features. The two aggregates are exactly ``models/har.py``'s ``RV_w`` and
``RV_m`` — mean of RV, then logged, not mean of log RV — so a HAR-equivalent
linear function of these features exists and the ensemble is being asked
whether the nonlinearity and the extra lags buy anything over Corsi (2009).
Logs for the same reason HAR uses them: RV is positive and right-skewed, and
an L2 objective on the level would be dominated by a handful of crisis days.

**Lag semantics under D-018 compaction.** ``t - 1`` here is a *position* in
the series handed to ``fit``, not a calendar day. On a compacted panel series
(``volbench.compaction``, the study's default) days whose variance target was
unmeasurable are dropped from that series, so "lag 1" can span two or more
calendar days, and the 22-lag window can span more than 22. The alternative
would be imputing a variance that was never observed, which is worse; but the
regressors are then "the previous 22 measured days", and any statement about
the model's memory in calendar time has to say so. On a series with no invalid
days — the toy fixture, and every series in the panel except HSI, TWSE, CAC
and NKX — position and calendar coincide exactly.

TEMPORAL INTEGRITY (CLAUDE.md rule 1) — the reason this module is written the
way it is
==========================================================================

Every feature is a function of ``rv[:t+1]`` and nothing else, and every
feature is computed **from the window handed to fit/update**, never from the
full series. Concretely:

- ``_design_matrix`` builds row ``t`` from ``rv[t - 21 : t + 1]`` only, with
  target ``log rv[t + 1]``. Rows start at ``t = 21`` so every window is full;
  there are no truncated or back-filled rows, matching HAR's strict-window
  rule.
- There is **no scaling, centring or normalization anywhere**. That is
  deliberate: the classic leak in a boosted-tree forecasting pipeline is a
  scaler fitted on the whole series (or on train+test) and then applied per
  window. Trees are invariant to monotone rescaling of a feature, so a scaler
  would buy nothing and could only cost temporal integrity. Its absence is
  the invariant, and ``tests/test_models_lgbm.py::TestLeakage`` pins it: a
  window's design matrix is bit-identical whether the array it came from
  continues afterwards or not.
- No early stopping, and therefore no validation split. ``num_boost_round``
  is a fixed hyperparameter in ``spec()``. Early stopping needs held-out
  data; the only data that could hold out is inside the training window, and
  a *random* split of a time series into train/validation is itself a leak
  (tomorrow's neighbours predict today). Fixed rounds sidestep the question
  rather than answering it badly.
- The out-of-fold folds the smearing factor is read off (below) are
  **chronological and expanding**, never random, for exactly that reason:
  block ``k`` is predicted by a booster trained on blocks ``0..k-1`` only, so
  no residual is ever produced by a model that saw its own row or any later
  one. ``tests/test_models_lgbm.py::TestOutOfFoldFoldsAreCausal`` is the
  canary: corrupt every design row from one fold boundary onward and the
  residuals of the fold before it must not move by one bit.

Retransformation
================

The model forecasts ``E[log RV]``; the benchmark needs a variance, i.e. a
mean. Same two arms as ``models/sf.py``, same default, both documented with
their evidence in ``volbench.models._rv``:

- ``retransform="smearing"`` (DEFAULT): Duan (1983) — ``exp(mu) * mean(exp(e))``
  over the fit window's residuals. **Which** residuals is the separate,
  config-hashed ``smearing_residuals`` choice below; it defaults to
  ``"out_of_fold"``.
- ``retransform="gaussian"``: ``exp(mu + sigma^2/2)`` with ``sigma^2`` the
  **in-sample** residual variance — exactly what ``models/har.py`` does, kept
  as the like-for-like comparison arm, and therefore deliberately unaffected
  by ``smearing_residuals``: an arm whose whole purpose is to be HAR's
  estimator on this model's residuals must keep HAR's construction.

Both factors are estimated once, at the scheduled fit, and are one-step
quantities reused at every horizon (see ``_rv``'s "Horizon caveat").

**Which residuals the factor is read off —** ``smearing_residuals``, and the
defect it exists to fix
-------------------------------------------------------------------------

Duan's estimator wants residuals that behave like draws from the forecast-
error distribution. HAR is an OLS with four parameters, so its in-sample
residuals are close to that. A boosted ensemble's are not: capacity shrinks
them, and a shrunken residual set drives the correction factor toward 1,
which quietly turns the variance forecast back into a *median* forecast.
Measured on the toy fixture at 500-observation windows, LightGBM's stock
shape (300 rounds, 31/15 leaves, ``min_data_in_leaf=20``) puts the in-sample
log-space residual variance at 0.015 against a realized one-step forecast-
error variance of 0.42 — a 28x understatement, and a smearing factor of 1.008
where HAR's is 1.207.

Two things bound it, and only the second one removes it:

``in_sample``
    The original construction: ``mean(exp(e))`` over the rows the booster
    trained on. It is kept as a named, config-hashed arm because it is what
    the deliberately small ensemble below was sized for — depth-2 trees,
    ``min_data_in_leaf`` at ~12% of the training rows, ``lambda_l2 = 5`` —
    and because the size of its bias is a measurement this package reports.
    ``tests/test_models_lgbm.py::TestRetransformation::
    test_the_ensemble_does_not_memorize_its_own_residuals`` is its regression
    guard: raise the capacity on that arm and it fails, rather than the
    correction silently going away.

``out_of_fold`` (DEFAULT)
    Expanding chronological folds inside the training window
    (:func:`out_of_fold_residuals`): the design rows are cut into
    ``oof_folds`` contiguous blocks and block ``k`` is predicted by a booster
    trained on blocks ``0..k-1`` only. The residuals are then genuine
    one-step-ahead errors of the same estimator at a smaller sample size, not
    fitted residuals, which is what Duan's iid-draws condition asks for.

**The measurement that settles the default** (``docs/P3_LGBM_SMEARING_AUDIT.md``,
all 2,366 refit origins of the primary grid, 11 assets, 21 years of index
data). At the panel median the in-sample factor is **1.371** where the grid's
own realized one-step forecast errors imply **1.678** — every stored ``lgbm``
variance sat ~17.9 % below the retransformation the theory asks for, with the
same sign on all eleven assets and a ratio of 5.57 against 1.37 inside the
COVID window. The out-of-fold factor is **1.703** against that same realized
1.678: a 1.5 % overshoot, in the direction fewer training rows predicts.

**A limitation this does not fix, recorded rather than worked around.** The
100-round cap is binding on the *training* loss — MSE falls monotonically
from 0.78 at 25 rounds to 0.24 at 800 with no plateau, and the in-sample
factor falls with it from 1.49 to 1.13 (audit §4). The capacity choice and
the in-sample factor's collapse are one mechanism, so the numbers above must
not be read as a property of the retransformation alone. Raising the cap
would change the model rather than fix the estimator, and is deliberately not
done here.

Re-conditioning between refits — ``SupportsUpdate`` IS implemented, exactly
==========================================================================

Refreshing the features under fixed trees is not an approximation of
re-conditioning, it *is* re-conditioning: the booster is a deterministic
function from a feature row to a prediction, so moving the RV buffer to the
end of a later window and re-reading the lags gives precisely the forecast
that a model with those parameters and that information set should make.
Nothing is re-estimated — not the trees, not the smearing factor (whichever
residuals it was read off), not the residual variance. ``update`` on the
fit's own window reproduces the fit.

Determinism (CLAUDE.md rule 3)
==============================

LightGBM is the one model in this package with real determinism hazards:
row/column subsampling draw from an RNG, and multi-threaded histogram
construction can reorder floating-point sums. All of it is pinned —
``deterministic=True`` (which requires ``force_row_wise=True``, set here),
a single fixed ``seed`` seeding every sub-seed, and ``num_threads=1``. Every
one of those is in ``spec()`` and therefore in the ``config_hash``, and
``tests/test_models_lgbm.py::TestDeterminism`` asserts a bit-identical
forecast from the same window twice, and an identical serialized model.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from volbench.dist import Distribution, Normal
from volbench.models._rv import (
    Retransform,
    gaussian_factor,
    smearing_factor,
    validated_rv,
    variance_from_log,
)

# Typing only: lightgbm lives in the optional ``classical`` extra and this
# module is re-exported from ``volbench.models``, so the runtime import happens
# inside ``fit`` and ``import volbench`` never needs it (Phase-2 integration).
if TYPE_CHECKING:
    import lightgbm as lgb

__all__ = [
    "DEFAULT_OOF_FOLDS",
    "FittedLightGBMRV",
    "LightGBMRV",
    "SmearingResiduals",
    "out_of_fold_residuals",
]

#: Which residuals Duan's smearing factor is estimated from. A named,
#: config-hashed option rather than a silent default, for the same reason
#: ``retransform`` is one (``volbench.models._rv``): the two produce
#: numerically different variance forecasts from one booster.
SmearingResiduals = Literal["in_sample", "out_of_fold"]

#: Contiguous, chronological blocks the training window is cut into for the
#: out-of-fold factor. Five leaves the first block as training-only and
#: predicts the other four, so 80 % of the window's rows contribute an
#: out-of-fold residual — the setting docs/P3_LGBM_SMEARING_AUDIT.md measured
#: 1.703 against a realized 1.678 at.
DEFAULT_OOF_FOLDS: Final = 5

#: Lags 1..22 of log RV. 22 is HAR's monthly window, so the lag block and the
#: monthly aggregate see the same information set.
_MAX_LAG: Final = 22
#: HAR's weekly and monthly aggregation windows (models/har.py).
_W_WINDOW: Final = 5
_M_WINDOW: Final = 22
_N_FEATURES: Final = _MAX_LAG + 2

#: Rows needed before an ensemble is worth fitting at all: a full 22-day
#: history plus enough targets that the default ``min_data_in_leaf`` (60) can
#: bind on more than a single split.
_MIN_ROWS: Final = 150
_MIN_TRAIN: Final = _M_WINDOW + _MIN_ROWS

FEATURE_NAMES: Final = tuple(
    [f"log_rv_lag{k}" for k in range(1, _MAX_LAG + 1)]
    + [f"log_mean_rv_{_W_WINDOW}", f"log_mean_rv_{_M_WINDOW}"]
)


def _feature_row(rv: NDArray[np.float64], t: int) -> NDArray[np.float64]:
    """Features at time ``t``, from ``rv[t - 21 : t + 1]`` and nothing else.

    ``rv[t]`` is "lag 1" — the most recent observation available when
    forecasting ``t + 1`` — matching HAR's ``RV_d``.

    The bounds check is not defensive padding, it is the leakage guard. With
    ``t < _M_WINDOW - 1`` the lag slice's start index goes negative, and numpy
    reads a negative start as ``len(rv) + start`` — so instead of raising, the
    row silently comes back the wrong length, built from a slice chosen by the
    array's *total length* rather than by ``t``. No caller can reach that today
    (``fit`` requires a much longer window, ``update`` validates 22 and
    ``_design_matrix`` starts at ``t = _M_WINDOW - 1``), and this makes sure
    the next one cannot either.
    """
    if t < _M_WINDOW - 1 or t >= rv.size:
        raise ValueError(
            f"feature row at t={t} needs a full {_M_WINDOW}-day history inside a "
            f"series of length {rv.size}"
        )
    lags = np.log(rv[t - _MAX_LAG + 1 : t + 1][::-1])
    weekly = math.log(float(np.mean(rv[t - _W_WINDOW + 1 : t + 1])))
    monthly = math.log(float(np.mean(rv[t - _M_WINDOW + 1 : t + 1])))
    return np.concatenate([lags, [weekly, monthly]]).astype(np.float64)


def _design_matrix(
    rv: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(X, y)`` over the window, strict full lag windows only.

    Row ``i`` predicts ``log rv[t + 1]`` from information dated ``t`` or
    earlier. The loop bounds are the whole leakage argument: ``t`` runs from
    ``_M_WINDOW - 1`` (the first index with a full 22-day history) to
    ``n - 2`` (the last index whose successor exists inside this window).
    """
    n = rv.size
    rows = [_feature_row(rv, t) for t in range(_M_WINDOW - 1, n - 1)]
    targets = [math.log(float(rv[t + 1])) for t in range(_M_WINDOW - 1, n - 1)]
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def out_of_fold_residuals(
    config: LightGBMRV,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    folds: int = DEFAULT_OOF_FOLDS,
) -> NDArray[np.float64]:
    """Residuals from expanding chronological folds inside one training window.

    The design rows are cut into ``folds`` contiguous blocks in their own
    order — never a random split, which on a time series lets tomorrow's
    neighbours predict today (``docs/design.md``) — and block ``k`` is
    predicted by a booster trained on blocks ``0..k-1`` only. Block ``0``
    yields no residual, there being nothing before it; that is the
    unavoidable price of keeping the fold causal, and it is why the estimate
    trains on 20-80 % of the window rather than all of it.

    **The causal boundary, stated arithmetically**, because it is the one
    thing a corruption canary cannot see. Design row ``i`` reads
    ``rv[i : i + 22]`` and predicts ``rv[i + 22]``, so a booster trained on
    rows ``0..train_end-1`` has seen targets up to ``rv[train_end + 21]`` —
    exactly the last observation known at the origin of row ``train_end``,
    whose own target is ``rv[train_end + 22]``. Causal with no gap, and no
    gap needed. ``tests/test_models_lgbm.py::TestOutOfFoldFoldsAreCausal``
    pins both halves.

    Raises ``ValueError`` when no fold can be formed rather than silently
    returning an empty array: an empty residual set would fall through to a
    factor that is not the one ``spec()`` names.
    """
    import lightgbm as lgb

    if folds < 2:
        raise ValueError("oof_folds must be >= 2")
    n = int(y.size)
    edges = np.linspace(0, n, folds + 1).astype(int)
    out: list[NDArray[np.float64]] = []
    for k in range(1, folds):
        train_end, block_end = int(edges[k]), int(edges[k + 1])
        if train_end < 2 or block_end <= train_end:
            continue
        dataset = lgb.Dataset(
            x[:train_end],
            label=y[:train_end],
            feature_name=list(FEATURE_NAMES),
            free_raw_data=False,
        )
        booster = lgb.train(config._params(), dataset, num_boost_round=config.num_boost_round)
        pred = np.asarray(
            booster.predict(x[train_end:block_end]), dtype=np.float64
        ).reshape(-1)
        out.append(y[train_end:block_end] - pred)
    if not out:
        raise ValueError(
            f"a window of {n} design rows is too short to form {folds} causal folds"
        )
    return np.concatenate(out)


@dataclass(frozen=True, eq=False)
class FittedLightGBMRV:
    """A trained booster plus the RV buffer its next forecast reads.

    ``eq=False``: numpy-array and Booster fields make the generated ``__eq__``
    raise (the trap documented on ``Origin`` in splitter.py).

    ``booster`` is shared by reference with everything ``update`` derives from
    this object. LightGBM's ``Booster.predict`` does not mutate the model, so
    that is safe; ``tests/test_models_lgbm.py`` asserts it rather than
    assuming it.
    """

    config: LightGBMRV
    booster: lgb.Booster
    #: The last ``_M_WINDOW`` realized variances — the regressors of the next
    #: forecast. Moves with ``update``; nothing else does.
    buffer: NDArray[np.float64]
    #: Duan's smearing factor over the residuals ``config.smearing_residuals``
    #: names — out-of-fold by default, in-sample on the other arm.
    smear: float
    #: **In-sample** residual variance in log space, for the ``gaussian`` arm
    #: and for the capacity guard. In-sample on both arms by construction:
    #: the ``gaussian`` arm exists to be HAR's estimator (module docstring).
    resid_var: float

    @property
    def name(self) -> str:
        return self.config.name

    def spec(self) -> dict[str, Any]:
        return self.config.spec()

    def _factor(self) -> float:
        if self.config.retransform == "smearing":
            return self.smear
        return gaussian_factor(self.resid_var)

    def predict(self, h: int) -> Distribution:
        """Forecast the variance ``h`` steps past the buffer's end.

        For ``h > 1`` this iterates the way ``models/har.py`` does: each
        step's retransformed point forecast is appended to the buffer as if
        realized and the feature row is rebuilt. The trees never change.
        """
        if h < 1:
            raise ValueError("h must be >= 1")
        factor = self._factor()
        buf = self.buffer.copy()
        rv_hat = 0.0
        for _ in range(h):
            x = _feature_row(buf, buf.size - 1).reshape(1, _N_FEATURES)
            mu = float(np.asarray(self.booster.predict(x), dtype=np.float64).reshape(-1)[0])
            rv_hat = variance_from_log(mu, factor)
            buf = np.append(buf[1:], rv_hat)
        return Normal(mu=0.0, sigma=math.sqrt(rv_hat))

    def update(self, train: NDArray[np.float64]) -> FittedLightGBMRV:
        """Refresh the feature buffer under the fitted trees.

        Exact, not approximate: the booster is a deterministic map from a
        feature row to a prediction, so re-reading the lags at a later origin
        is precisely the forecast this parameterization implies there. The
        trees, the smearing factor and the residual variance are those of the
        last scheduled refit and do not move.
        """
        rv = validated_rv(train, minimum=_M_WINDOW)
        return dataclasses.replace(self, buffer=rv[-_M_WINDOW:].copy())


@dataclass(frozen=True)
class LightGBMRV:
    """LightGBM on lagged log-RV, retransformed to a variance forecast.

    Defaults are sized for the ~500-observation rolling windows this package
    backtests on (docs/research_design.md): a shallow, strongly regularized
    ensemble, because ~480 training rows and 24 correlated features is a
    setting where a stock GBM memorizes — and here memorizing does not merely
    overfit, it disables the log-to-variance retransformation. See the module
    docstring for the measurement behind these numbers.
    """

    retransform: Retransform = "smearing"
    #: Which residuals the ``smearing`` arm's factor is estimated from. The
    #: default is the out-of-fold construction; ``"in_sample"`` is the
    #: original one, kept as a named arm (module docstring). Inert on the
    #: ``gaussian`` arm, and recorded in ``spec()`` regardless, because a
    #: field that decides a number must decide the identity of every run.
    smearing_residuals: SmearingResiduals = "out_of_fold"
    #: Chronological blocks the out-of-fold estimate cuts the window into.
    oof_folds: int = DEFAULT_OOF_FOLDS
    num_boost_round: int = 100
    learning_rate: float = 0.05
    #: 4 leaves == depth-2 trees. Two-way interactions among lagged log-RVs
    #: are the structure HAR cannot express; deeper trees on ~480 rows fit
    #: noise, and the residual shrinkage that follows disables the
    #: retransformation (module docstring).
    num_leaves: int = 4
    min_data_in_leaf: int = 60
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l2: float = 5.0
    max_bin: int = 255
    seed: int = 20260825
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.retransform not in ("smearing", "gaussian"):
            raise ValueError("retransform must be 'smearing' or 'gaussian'")
        if self.smearing_residuals not in ("in_sample", "out_of_fold"):
            raise ValueError("smearing_residuals must be 'in_sample' or 'out_of_fold'")
        if self.oof_folds < 2:
            raise ValueError("oof_folds must be >= 2")
        if self.num_boost_round < 1:
            raise ValueError("num_boost_round must be >= 1")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be >= 2")
        if not 0.0 < self.feature_fraction <= 1.0:
            raise ValueError("feature_fraction must lie in (0, 1]")
        if not 0.0 < self.bagging_fraction <= 1.0:
            raise ValueError("bagging_fraction must lie in (0, 1]")
        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")

    @property
    def name(self) -> str:
        return f"lightgbm_rv-{self.retransform}"

    def _params(self) -> dict[str, Any]:
        """The exact parameter dict handed to LightGBM.

        ``deterministic`` requires one of ``force_row_wise`` /
        ``force_col_wise``; without it LightGBM picks a histogram construction
        strategy by timing the machine it is on, which is not reproducible
        across machines. Both are pinned, as is ``num_threads`` (thread count
        changes the order of floating-point reductions) and ``seed`` (which
        seeds bagging, feature sub-sampling and dataset binning alike).
        """
        return {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "max_bin": self.max_bin,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": self.num_threads,
            "verbosity": -1,
        }

    def spec(self) -> dict[str, Any]:
        return {
            "model": "lightgbm_rv",
            "backend": "lightgbm",
            "target": "log_rv",
            "max_lag": _MAX_LAG,
            "w_window": _W_WINDOW,
            "m_window": _M_WINDOW,
            "n_features": _N_FEATURES,
            "num_boost_round": self.num_boost_round,
            "retransform": self.retransform,
            "smearing_residuals": self.smearing_residuals,
            "oof_folds": self.oof_folds,
            "min_train": _MIN_TRAIN,
            **self._params(),
        }

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedLightGBMRV:
        import lightgbm as lgb

        rv = validated_rv(train, minimum=_MIN_TRAIN)
        x, y = _design_matrix(rv)
        dataset = lgb.Dataset(
            x, label=y, feature_name=list(FEATURE_NAMES), free_raw_data=False
        )
        booster = lgb.train(
            self._params(), dataset, num_boost_round=self.num_boost_round
        )
        resid = y - np.asarray(booster.predict(x), dtype=np.float64).reshape(-1)
        smear_resid = (
            out_of_fold_residuals(self, x, y, self.oof_folds)
            if self.smearing_residuals == "out_of_fold"
            else resid
        )
        return FittedLightGBMRV(
            config=self,
            booster=booster,
            buffer=rv[-_M_WINDOW:].copy(),
            smear=smearing_factor(smear_resid),
            resid_var=float(np.mean(resid * resid)),
        )
