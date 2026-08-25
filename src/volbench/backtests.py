"""VaR / ES backtests on volbench result rows: Kupiec, Christoffersen, FZ0.

Consumes the hit indicators, VaR quantiles and realized returns that
:func:`volbench.evaluate.run_backtest` records (``hit_<level>``,
``var_<level>``, ``realized_return``) and never touches the scored path.

Three tools (docs/metrics_reference.md, "VaR / ES backtesting"):

- :func:`kupiec_pof` — Kupiec's (1995) proportion-of-failures likelihood
  ratio test of unconditional coverage ``E[H_t] = alpha``.
- :func:`christoffersen` — Christoffersen's (1998) independence test
  against a first-order Markov alternative, and the conditional-coverage
  test that combines it with unconditional coverage.
- :func:`fz0_loss` — the zero-degree-homogeneous Fissler-Ziegel loss in the
  form used by Patton, Ziegel & Chen (2019, eq. 6), to *score and rank*
  joint (VaR, ES) forecasts rather than only to test them. Since v0.4.0
  result rows carry the ES forecast alongside the VaR quantile
  (``es_<level>``, written by the evaluator from the predictive
  distribution), so :func:`var_backtest` scores FZ0 without being handed
  anything extra; :func:`expected_shortfall` remains available for a
  :class:`~volbench.dist.Distribution` in hand, and for reading rows
  produced before that column existed.

Every test result carries ``n`` and ``expected_hits = n * level`` and a
``small_sample`` flag; below :data:`MIN_EXPECTED_HITS` expected exceedances
a :class:`SmallSampleWarning` is issued as well. Kupiec (1995) already
documents that at the 1% level with a few hundred observations the POF test
has almost no power to detect a mis-stated coverage; the API surfaces that
rather than hiding it behind a p-value.

Conventions
-----------
- ``level`` is the tail probability of the (lower-tail) VaR, e.g. ``0.01``,
  and the return-side sign convention is volbench's: VaR and ES are
  quantities of the *return* distribution, negative in the lower tail.
- A hit is ``H_t = 1{r_t < VaR_t}`` — strict, exactly as ``evaluate.py``
  records ``hit_<level>``. The FZ0 loss uses ``1{Y <= v}`` as written in
  PZC (2019); for continuous returns the two agree almost surely.
- **NaN policy.** Hits that are NaN (unscorable origins) are excluded and
  counted in ``n_dropped``; the Markov transition counts only pair
  observations that are *adjacent in the original sequence*, so a dropped
  origin never manufactures a transition across the gap.

References
----------
Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk
measurement models. *Journal of Derivatives* 3(2), 73-84.

Christoffersen, P. F. (1998). Evaluating interval forecasts. *International
Economic Review* 39(4), 841-862.

Fissler, T. & Ziegel, J. F. (2016). Higher order elicitability and Osband's
principle. *Annals of Statistics* 44(4), 1680-1707.

Patton, A. J., Ziegel, J. F. & Chen, R. (2019). Dynamic semiparametric models
for expected shortfall (and Value-at-Risk). *Journal of Econometrics* 211(2),
388-413.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special, stats  # type: ignore[import-untyped]

from volbench.dist import Distribution
from volbench.results import array_digest, config_hash, package_version

__all__ = [
    "MIN_EXPECTED_HITS",
    "ChristoffersenResult",
    "KupiecResult",
    "MissingPolicy",
    "SmallSampleWarning",
    "VaRBacktest",
    "christoffersen",
    "expected_shortfall",
    "fz0_loss",
    "hit_indicators",
    "kupiec_pof",
    "var_backtest",
]

#: Below this many *expected* exceedances (``n * level``) the chi-square
#: approximation to the LR tests is poor and their power against realistic
#: mis-coverage is negligible (Kupiec 1995, Table 1: at 1% with 255 days —
#: 2.55 expected hits — the POF test cannot separate 1% from 2-3% coverage).
#: A rule of thumb, not a theorem; the counts are always reported so a
#: reader can judge.
MIN_EXPECTED_HITS: Final = 10.0

#: Which result rows :func:`var_backtest` treats as missing; mirrors
#: :data:`volbench.inference.MissingPolicy`.
MissingPolicy = Literal["flagged", "score"]


class SmallSampleWarning(UserWarning):
    """A VaR backtest was run with fewer than :data:`MIN_EXPECTED_HITS` expected hits."""


# --------------------------------------------------------------------------
# hits
# --------------------------------------------------------------------------


def hit_indicators(returns: object, var: object) -> NDArray[np.float64]:
    """``H_t = 1{r_t < VaR_t}`` as a float array, NaN where either input is non-finite.

    Strict inequality, matching the ``hit_<level>`` columns ``evaluate.py``
    records.
    """
    r = np.asarray(returns, dtype=np.float64)
    v = np.asarray(var, dtype=np.float64)
    if r.shape != v.shape:
        raise ValueError(f"returns {r.shape} and var {v.shape} must have the same shape")
    valid = np.isfinite(r) & np.isfinite(v)
    out = np.full(r.shape, np.nan)
    out[valid] = (r[valid] < v[valid]).astype(np.float64)
    return out


def _clean_hits(hits: object) -> tuple[NDArray[np.float64], NDArray[np.bool_], int]:
    """Validate a hit sequence: values in {0, 1} or NaN; returns (hits, valid mask, n_dropped)."""
    h = np.asarray(hits, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError(f"hits must be 1-D, got shape {h.shape}")
    valid = np.isfinite(h)
    if not np.all(np.isin(h[valid], (0.0, 1.0))):
        raise ValueError("hits must be 0/1 indicators (NaN allowed for unscorable origins)")
    return h, valid, int(h.size - int(valid.sum()))


def _check_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly inside (0, 1), got {level}")
    return float(level)


def _small_sample(n: int, level: float, *, warn: bool, what: str) -> tuple[float, bool]:
    expected = n * level
    small = expected < MIN_EXPECTED_HITS
    if small and warn:
        warnings.warn(
            f"{what}: n={n} at level {level:g} gives {expected:.2f} expected exceedances "
            f"(< {MIN_EXPECTED_HITS:g}); the likelihood-ratio tests have little power and the "
            "chi-square approximation is rough at this sample size (Kupiec 1995)",
            SmallSampleWarning,
            stacklevel=3,
        )
    return float(expected), bool(small)


def _xlogy(x: float, y: float) -> float:
    """``x * log(y)`` with the ``0 * log(0) = 0`` convention of the LR tests."""
    return float(special.xlogy(x, y))


# --------------------------------------------------------------------------
# Kupiec (1995) proportion of failures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KupiecResult:
    """Kupiec POF test. ``n`` observations used, ``n_hits`` exceedances,
    ``expected_hits = n * level``; ``small_sample`` is set below
    :data:`MIN_EXPECTED_HITS`. ``n_dropped`` counts NaN hits excluded."""

    level: float
    n: int
    n_hits: int
    expected_hits: float
    hit_rate: float
    lr: float
    p_value: float
    n_dropped: int
    small_sample: bool


def kupiec_pof(hits: object, level: float, *, warn: bool = True) -> KupiecResult:
    """Kupiec's (1995) proportion-of-failures test of ``E[H_t] = level``.

    With ``x`` exceedances in ``n`` observations and nominal failure
    probability ``p = level``, the likelihood ratio of the binomial
    restriction ``pi = p`` against the unrestricted ``pi_hat = x/n`` is

        ``LR_POF = -2 ln[ (1-p)^{n-x} p^x ] + 2 ln[ (1 - x/n)^{n-x} (x/n)^x ]``,

    asymptotically ``χ²(1)`` under the null (Kupiec 1995; the same form as
    Christoffersen's 1998 unconditional-coverage test). ``0 · ln 0 = 0``, so
    ``x = 0`` gives ``-2n ln(1 - p)`` and ``x = n`` gives ``-2n ln p``.

    The test has notoriously low power at small ``level * n`` — Kupiec's
    own Table 1 — so the result always reports ``n`` and ``expected_hits``,
    and a :class:`SmallSampleWarning` is raised below
    :data:`MIN_EXPECTED_HITS` expected exceedances (``warn=False`` silences
    it; the ``small_sample`` flag is set regardless).
    """
    level = _check_level(level)
    h, valid, n_dropped = _clean_hits(hits)
    n = int(valid.sum())
    if n < 1:
        raise ValueError("need at least one usable hit indicator")
    x = int(h[valid].sum())
    expected, small = _small_sample(n, level, warn=warn, what="kupiec_pof")
    ll_null = _xlogy(n - x, 1.0 - level) + _xlogy(x, level)
    ll_alt = _xlogy(n - x, 1.0 - x / n) + _xlogy(x, x / n)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return KupiecResult(
        level=level,
        n=n,
        n_hits=x,
        expected_hits=expected,
        hit_rate=x / n,
        lr=lr,
        p_value=float(stats.chi2.sf(lr, df=1)),
        n_dropped=n_dropped,
        small_sample=small,
    )


# --------------------------------------------------------------------------
# Christoffersen (1998) independence and conditional coverage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChristoffersenResult:
    """Christoffersen's Markov tests on a hit sequence.

    ``n00 .. n11`` are the transition counts (``n_ij``: state ``i`` at
    ``t-1`` followed by state ``j`` at ``t``) over the ``n_transitions``
    adjacent pairs of usable observations. ``lr_uc``, ``lr_ind`` and
    ``lr_cc`` are the unconditional-coverage, independence and
    conditional-coverage likelihood ratios, all computed conditional on the
    first observation so that ``lr_cc == lr_uc + lr_ind`` holds exactly
    (``lr_uc`` therefore differs from :func:`kupiec_pof`'s full-sample
    statistic by one observation). ``n`` and ``n_hits`` count every usable
    observation; ``expected_hits = n * level``.
    """

    level: float
    n: int
    n_hits: int
    expected_hits: float
    n_transitions: int
    n00: int
    n01: int
    n10: int
    n11: int
    pi01: float
    pi11: float
    lr_uc: float
    lr_ind: float
    lr_cc: float
    p_uc: float
    p_ind: float
    p_cc: float
    n_dropped: int
    small_sample: bool


def christoffersen(hits: object, level: float, *, warn: bool = True) -> ChristoffersenResult:
    """Christoffersen's (1998) independence and conditional-coverage tests.

    Embeds the iid-Bernoulli null in a first-order Markov chain for the hit
    sequence with transition probabilities ``pi_ij = P(H_t = j | H_{t-1} =
    i)``. With transition counts ``n_ij`` and likelihood

        ``L(pi01, pi11) = (1-pi01)^{n00} pi01^{n01} (1-pi11)^{n10} pi11^{n11}``,

    estimated by ``pi_hat01 = n01/(n00+n01)``, ``pi_hat11 = n11/(n10+n11)`` and, under
    independence, ``pi_hat = (n01+n11)/(n00+n01+n10+n11)``:

    - independence: ``LR_ind = -2 ln[ L(pi_hat, pi_hat) / L(pi_hat01, pi_hat11) ]  ~ χ²(1)``;
    - unconditional coverage: ``LR_uc = -2 ln[ L(p, p) / L(pi_hat, pi_hat) ] ~ χ²(1)``
      with ``p = level``;
    - conditional coverage: ``LR_cc = -2 ln[ L(p, p) / L(pi_hat01, pi_hat11) ] =
      LR_uc + LR_ind ~ χ²(2)``.

    All three condition on the first observation (Christoffersen 1998;
    Berkowitz, Christoffersen & Pelletier 2011 restate the same likelihood),
    which is what makes the decomposition exact. ``0 · ln 0 = 0`` throughout,
    so a sequence with no exceedances has ``LR_ind = 0`` and an undefined
    ``pi_hat11`` contributes nothing — the MATLAB/`rugarch` convention.

    Transitions are counted only between observations adjacent in the
    original sequence: a NaN hit removes its two neighbouring transitions
    rather than splicing the observations around it together. "Adjacent"
    means consecutive *rows*: for result rows from a splitter with ``step >
    1`` the neighbours are ``step`` days apart and the test concerns
    dependence at that spacing, not at one day.
    """
    level = _check_level(level)
    h, valid, n_dropped = _clean_hits(hits)
    n = int(valid.sum())
    if n < 2:
        raise ValueError("need at least two usable hit indicators for transition counts")
    both = valid[1:] & valid[:-1]
    prev = h[:-1][both]
    cur = h[1:][both]
    n_transitions = int(both.sum())
    if n_transitions < 1:
        raise ValueError("no adjacent pair of usable hit indicators: cannot count transitions")
    n00 = int(np.sum((prev == 0.0) & (cur == 0.0)))
    n01 = int(np.sum((prev == 0.0) & (cur == 1.0)))
    n10 = int(np.sum((prev == 1.0) & (cur == 0.0)))
    n11 = int(np.sum((prev == 1.0) & (cur == 1.0)))
    x = int(h[valid].sum())
    expected, small = _small_sample(n, level, warn=warn, what="christoffersen")

    pi01 = n01 / (n00 + n01) if n00 + n01 > 0 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 > 0 else 0.0
    pi = (n01 + n11) / n_transitions
    ll_level = _xlogy(n00 + n10, 1.0 - level) + _xlogy(n01 + n11, level)
    ll_pi = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    ll_markov = (
        _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    )
    lr_uc = max(-2.0 * (ll_level - ll_pi), 0.0)
    lr_ind = max(-2.0 * (ll_pi - ll_markov), 0.0)
    lr_cc = lr_uc + lr_ind
    return ChristoffersenResult(
        level=level,
        n=n,
        n_hits=x,
        expected_hits=expected,
        n_transitions=n_transitions,
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        pi01=pi01,
        pi11=pi11,
        lr_uc=lr_uc,
        lr_ind=lr_ind,
        lr_cc=lr_cc,
        p_uc=float(stats.chi2.sf(lr_uc, df=1)),
        p_ind=float(stats.chi2.sf(lr_ind, df=1)),
        p_cc=float(stats.chi2.sf(lr_cc, df=2)),
        n_dropped=n_dropped,
        small_sample=small,
    )


# --------------------------------------------------------------------------
# FZ0 loss (Fissler & Ziegel 2016; Patton, Ziegel & Chen 2019, eq. 6)
# --------------------------------------------------------------------------


def fz0_loss(returns: object, var: object, es: object, level: float) -> NDArray[np.float64]:
    """Elementwise FZ0 loss of joint (VaR, ES) forecasts, PZC (2019) eq. (6).

    Verified against the paper (their eqs. 6 and 42, identical):

        ``L_FZ0(Y, v, e; alpha) = -(1 / (alpha e)) · 1{Y ≤ v} · (v - Y) + v/e + log(-e) - 1``,

    with ``Y`` the realized return, ``v`` the VaR forecast (the alpha-quantile of
    the return), ``e`` the ES forecast (the mean of the return below ``v``)
    and ``alpha = level``. It is the member of the Fissler-Ziegel (2016) family
    with ``G_1 = 0`` and ``G_2(e) = -1/e`` (antiderivative ``-log(-e)``), the unique
    choice — up to location and scale — whose loss *differences* are
    homogeneous of degree zero when VaR and ES are negative (PZC Prop. 1).
    The loss itself is not: ``L(kY, kv, ke) = L(Y, v, e) + log k`` for ``k >
    0`` (PZC, remark after eq. 42), which is pinned in the tests. Its
    expectation is minimized at the true (VaR, ES) — PZC Figure 2 — so
    averaging it over origins ranks ES forecasts consistently.

    Domain, from the paper: the ``log(-e)`` term requires ``e < 0`` — PZC
    assume ``ES_t < 0`` a.s. for ``alpha`` in the 1-10% range that matters in
    risk management — and consistency of any FZ loss requires ``e ≤ v``
    (their footnote 1). Both are enforced: a forecast with ``e >= 0`` or
    ``e > v`` raises rather than scoring, because the usual way to violate
    them is a sign-convention bug (passing loss-side, positive VaR/ES). A
    positive ``v`` with ``e < 0`` is accepted (PZC footnote 2: still a
    consistent FZ loss, with one shape parameter set to zero).

    Non-finite inputs give NaN at that position; callers count them.
    """
    level = _check_level(level)
    y, v, e = np.broadcast_arrays(
        np.asarray(returns, dtype=np.float64),
        np.asarray(var, dtype=np.float64),
        np.asarray(es, dtype=np.float64),
    )
    finite = np.isfinite(y) & np.isfinite(v) & np.isfinite(e)
    if np.any(e[np.isfinite(e)] >= 0.0):
        raise ValueError(
            "FZ0 needs ES forecasts strictly below zero (return-side sign convention: the "
            "lower-tail expected shortfall is negative); got a non-negative ES"
        )
    pair = np.isfinite(v) & np.isfinite(e)
    if np.any(e[pair] > v[pair]):
        raise ValueError(
            "FZ0 needs ES <= VaR (the mean below the quantile cannot exceed it); a forecast "
            "with ES > VaR is incoherent"
        )
    out = np.full(y.shape, np.nan)
    yy, vv, ee = y[finite], v[finite], e[finite]
    shortfall = np.where(yy <= vv, vv - yy, 0.0)
    out[finite] = -shortfall / (level * ee) + vv / ee + np.log(-ee) - 1.0
    return out


# --------------------------------------------------------------------------
# expected shortfall of a predictive distribution
# --------------------------------------------------------------------------


def expected_shortfall(dist: Distribution, level: float) -> float:
    """Lower-tail expected shortfall ``ES_alpha = alpha^{-1} ∫_0^alpha Q(u) du`` of a forecast.

    The mean of the return below its ``alpha``-quantile — the ``e`` that
    :func:`fz0_loss` scores, in the same return-side sign convention as the
    ``var_<level>`` columns (negative in the lower tail).

    Thin wrapper over :meth:`volbench.dist.Distribution.expected_shortfall`,
    which is where the arithmetic lives: closed forms on
    :class:`~volbench.dist.Normal` and :class:`~volbench.dist.StudentT`, the
    exact integral of the piecewise-linear quantile function on
    :class:`~volbench.dist.Empirical` and
    :class:`~volbench.dist.QuantileGrid`, and Gauss-Legendre quadrature
    otherwise.

    It moved there when ``evaluate.py`` began recording ``es_<level>`` columns
    at scoring time (D-018's sibling change, v0.4.0). Both this module and the
    evaluator need the same number, and the evaluator must not import the
    backtests — the dependency runs evaluation → results → distributions, and
    the backtests read the evaluator's *output*, never the other way round.
    Putting it on the distribution, next to ``variance`` and ``crps``, is the
    only placement where neither consumer imports the other. This name stays
    because it is part of the package's public surface.
    """
    return dist.expected_shortfall(_check_level(level))


# --------------------------------------------------------------------------
# from result rows
# --------------------------------------------------------------------------


def _level_tag(level: float) -> str:
    """Mirrors ``evaluate._level_tag`` — the result columns are named by it."""
    return f"{level:.10g}".replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class VaRBacktest:
    """Every backtest for one cell at one level, with the counts that qualify it.

    ``n`` is the number of usable origins, ``n_dropped`` how many were
    excluded (flagged ``missing_reason`` or NaN hit), ``expected_hits = n *
    level``. ``fz0_mean`` is the average FZ0 loss over the ``fz0_n`` origins
    with an ES forecast, or ``None`` when the cell carried no ``es_<level>``
    column and none was passed.
    """

    level: float
    n: int
    n_dropped: int
    n_hits: int
    expected_hits: float
    hit_rate: float
    small_sample: bool
    kupiec: KupiecResult
    christoffersen: ChristoffersenResult
    fz0_mean: float | None
    fz0_n: int
    config_hash: str


def var_backtest(
    frame: pd.DataFrame,
    level: float,
    *,
    es: object | None = None,
    policy: MissingPolicy = "flagged",
    warn: bool = True,
) -> VaRBacktest:
    """Kupiec, Christoffersen and (optionally) mean FZ0 for one result cell.

    ``frame`` holds the rows of **one** cell (one ``config_hash``: one
    model, asset, splitter and seed) and must contain ``hit_<level>``,
    ``var_<level>``, ``realized_return``, ``origin_index`` and
    ``missing_reason``; rows are put in ascending origin order first, since
    the independence test reads adjacency (consecutive origins: one day apart
    at ``step=1``, ``step`` days apart otherwise). Rows are excluded under
    ``policy`` — ``"flagged"`` (default) excludes any row with a
    ``missing_reason``, ``"score"`` only rows whose hit is NaN — and the
    exclusion is reported in ``n_dropped`` and on both test results.

    ``es`` supplies expected-shortfall forecasts aligned with ``frame``'s
    rows (an array, or the name of a column holding them; see
    :func:`expected_shortfall`). Left unset it defaults to the cell's **own**
    ``es_<level>`` column, which every row scored at v0.4.0 or later carries:
    that column is this model's ES at this level, computed from the same
    predictive distribution as ``var_<level>``, so it is the only ES that
    belongs in an FZ0 loss against these VaRs. On rows produced before the
    column existed there is nothing to read and ``fz0_mean`` stays ``None``;
    an ES is never approximated from ``forecast_var``, because that would
    presume a distributional family the row does not record.
    """
    level = _check_level(level)
    if policy not in ("flagged", "score"):
        raise ValueError(f"policy must be 'flagged' or 'score', got {policy!r}")
    tag = _level_tag(level)
    hit_col, var_col, es_col = f"hit_{tag}", f"var_{tag}", f"es_{tag}"
    required = [
        hit_col,
        var_col,
        "realized_return",
        "origin_index",
        "missing_reason",
        "config_hash",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"result frame is missing columns {missing}; was level {level:g} among the scored "
            "levels of this run?"
        )
    if frame.empty:
        raise ValueError("result frame is empty")
    hashes = sorted({str(h) for h in frame["config_hash"].unique()})
    if len(hashes) != 1:
        raise ValueError(
            f"var_backtest needs the rows of exactly one cell, got {len(hashes)} config hashes; "
            "filter to one model/asset/seed first"
        )
    if bool(frame["origin_index"].duplicated().any()):
        raise ValueError("duplicate origin_index rows in result frame (more than one horizon?)")

    es_values: NDArray[np.float64] | None
    if es is None:
        # The cell's own stored ES, when the run that produced these rows was
        # new enough to write it. Explicitly not a fallback to anything else.
        es_values = (
            np.asarray(frame[es_col].to_numpy(), dtype=np.float64)
            if es_col in frame.columns
            else None
        )
    elif isinstance(es, str):
        if es not in frame.columns:
            raise ValueError(f"es column {es!r} not in frame")
        es_values = np.asarray(frame[es].to_numpy(), dtype=np.float64)
    else:
        es_values = np.asarray(es, dtype=np.float64)
        if es_values.shape != (len(frame),):
            raise ValueError(
                f"es must have one value per row ({len(frame)}), got {es_values.shape}"
            )

    order = np.argsort(frame["origin_index"].to_numpy(), kind="stable")
    ordered = frame.iloc[order]
    hits = np.asarray(ordered[hit_col].to_numpy(), dtype=np.float64)
    returns = np.asarray(ordered["realized_return"].to_numpy(), dtype=np.float64)
    var = np.asarray(ordered[var_col].to_numpy(), dtype=np.float64)
    usable = np.isfinite(hits)
    if policy == "flagged":
        flagged = ordered["missing_reason"].fillna("").astype(str).str.len().to_numpy() > 0
        usable &= ~flagged
    hits = np.where(usable, hits, np.nan)
    n = int(usable.sum())
    n_dropped = int(len(frame) - n)
    if n < 2:
        raise ValueError(f"need at least two usable origins, got {n} (dropped {n_dropped})")

    expected, small = _small_sample(n, level, warn=warn, what="var_backtest")
    kupiec = kupiec_pof(hits, level, warn=False)
    markov = christoffersen(hits, level, warn=False)

    fz0_mean: float | None = None
    fz0_n = 0
    es_scored: NDArray[np.float64] | None = None
    if es_values is not None:
        # Excluded rows are masked out *before* scoring, not after. A row the
        # policy already dropped — a failed fit, an unscorable target — can
        # carry a degenerate ES, and FZ0's domain checks (ES < 0, ES <= VaR)
        # would otherwise reject the whole cell over a row that contributes
        # nothing to it. On rows that do count, those checks still fire: an
        # ES above its own VaR is an incoherent forecast, not a missing one.
        es_scored = np.where(usable, es_values[order], np.nan)
        losses = np.where(usable, fz0_loss(returns, var, es_scored, level), np.nan)
        fz0_n = int(np.isfinite(losses).sum())
        fz0_mean = float(np.nanmean(losses)) if fz0_n else math.nan

    digest = config_hash(
        {
            "method": "var_backtest",
            "cell": hashes[0],
            "level": level,
            "policy": policy,
            "hits_sha256": array_digest(hits),
            "returns_sha256": array_digest(returns),
            "var_sha256": array_digest(var),
            "es_sha256": None if es_scored is None else array_digest(es_scored),
            "package_version": package_version(),
        }
    )
    return VaRBacktest(
        level=level,
        n=n,
        n_dropped=n_dropped,
        n_hits=kupiec.n_hits,
        expected_hits=expected,
        hit_rate=kupiec.hit_rate,
        small_sample=small,
        kupiec=kupiec,
        christoffersen=markov,
        fz0_mean=fz0_mean,
        fz0_n=fz0_n,
        config_hash=digest,
    )
