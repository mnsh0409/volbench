"""PatchTST (Nie, Nguyen, Sinthong & Kalagnanam, ICLR 2023) — the trained deep-learning baseline.

Why PatchTST, and an assumption flagged for review
==================================================
The research design (model list, docs/research_design.md) wants one trained
DL baseline next to the zero-shot foundation models. PatchTST was chosen over
N-BEATS on the ASSUMPTION that architectural proximity is the more useful
comparison: Chronos-Bolt, TimesFM and Moirai are all patch-based transformers
(patch the series, embed, self-attend, project), so a small PatchTST trained
per window isolates "pretraining at scale" from "the architecture" in a way
an MLP-stack like N-BEATS would not. The price is that N-BEATS is the stronger
*univariate point-forecasting* baseline in several published comparisons; if
the reviewers' question is "can a trained net beat HAR" rather than "what
does pretraining buy", the choice should be revisited. The architecture is
small and fixed — a baseline, not a contender being optimized: there is no
tuning loop, and every hyperparameter is in ``spec()`` and hence in the
config hash.

Contract — same input as HAR, same output as every model
========================================================
``fit(train)`` takes a 1-D **realized-variance** window (daily units, finite,
strictly positive) and models ``log RV``. ``predict(h)`` returns
``Normal(mu=0, sigma=sqrt(vhat))`` over the next-period *return*, with
``vhat`` the variance forecast — the package convention (CLAUDE.md rule 2).

Retransformation: a net fit on ``log RV`` forecasts a conditional median in
levels, and a variance forecast is a mean. Duan's (1983) smearing estimate is
applied, as in the Phase-2 classical adapters: ``vhat = exp(mu_hat) *
mean_i(exp(e_i))`` with ``e_i`` the in-sample residuals of the training
windows. The factor is computed **per horizon** from that horizon's own
residuals (the net emits ``max_horizon`` direct outputs), so unlike HAR's
one-step correction it is not an approximation at ``h > 1``. HAR's Gaussian
``exp(sigma^2/2)`` correction is the like-for-like alternative; docs/M2_NOTES.md
records why the nonparametric factor is preferred on this target.

Architecture (fixed; every number below is in ``spec()``)
=========================================================
Channel-independent PatchTST for one series: the last ``lookback`` log-RV
values, instance-normalized (RevIN without affine — mean and std of *that*
input window only, so the transform is origin-local by construction), cut
into ``patch_len``-long patches at ``stride``, linearly embedded to
``d_model`` with a learned positional embedding, ``n_layers`` pre-norm
encoder layers (``n_heads`` heads, ``d_ff`` feed-forward, GELU, ``dropout``),
flattened and projected to ``max_horizon`` direct outputs. Attention is
explicit matmul + softmax rather than the fused kernels, for determinism.

Training budget — bounded and hashed
====================================
Adam (``lr``, ``weight_decay``), MSE in normalized log space, mini-batches of
``batch_size`` in a seeded random order, at most ``max_epochs`` epochs; early
stopping on the validation MSE with ``patience`` epochs of no improvement,
restoring the best weights. Validation is the chronologically **last**
``val_fraction`` of the training windows — never a random subset — and lies
entirely inside the fit window. What the budget actually consumed
(``epochs_run``, ``best_epoch``, ``best_val_mse``, ``stopped_early``) is in
the fitted ``spec()``.

Window handling — the leakage-check focus
=========================================
All training pairs are cut from the array ``fit`` was handed and nothing else:
input ``y[i : i+L]``, target ``y[i+L : i+L+H]`` for ``i = 0 .. n-L-H``. The
last target is ``y[n-1]``, the window's final observation, which is the
origin itself. The forecast input is ``y[n-L : n]``. No batch, no
normalization statistic and no validation split can reach past the origin
because none of them can reach past the end of the array.

``update()`` is deliberately NOT implemented
============================================
Re-conditioning a trained network without re-estimating it is not well
defined: the only state is the weights, and "conditioning on newer data"
would mean either refitting (which defeats the refit schedule) or sliding the
forecast input forward under frozen weights (which is a different, untested
model — the net was selected on targets up to the origin it was fit at).
PatchTST therefore runs **frozen between refits**: at every non-refit origin
the evaluator holds the forecast issued at the last scheduled fit, and the
``conditioned_through`` column records that origin (``== fit_origin``) on
every row. Compare it with the re-conditioned baselines with that in mind.

Determinism
===========
Fixed seed for weight init, batch order and dropout;
``torch.use_deterministic_algorithms(True)`` and cuDNN determinism for the
duration of the fit (restored afterwards); ``CUBLAS_WORKSPACE_CONFIG`` set at
import if unset; single device. Same window + same seed => bit-identical
forecast twice, on one device — pinned by test on CPU and on the GPU.
``device`` is where the computation runs, not what is computed, and is not
in ``spec()``. Across devices: with ``dropout=0`` CPU and CUDA fits agree to
float rounding (measured ~1e-8 relative); with dropout on, each device draws
its masks from its own RNG stream, so a CPU fit and a CUDA fit of the same
seed are two training realisations (a few percent apart on a 300-point
window). Results therefore reproduce *per device class*; the paper's runs
are all on the one GPU, and the tests pin both facts.

Dependency: torch (extra ``torch-cpu`` in CI, ``tsfm`` on the GPU box).
``volbench.models`` imports this module without torch; ``fit`` needs it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volbench.dist import Distribution, Normal

__all__ = ["FittedPatchTST", "PatchTST"]

# cuBLAS reads this when it creates its handles, which happens on the first
# CUDA op in the process; deterministic mode refuses cuBLAS calls otherwise.
# Set at import so any volbench process is covered before it touches CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _torch_version() -> str:
    try:
        return version("torch")
    except PackageNotFoundError:
        return "not-installed"


def _validated_rv(train: NDArray[np.float64], minimum: int) -> NDArray[np.float64]:
    rv = np.asarray(train, dtype=np.float64)
    if rv.ndim != 1 or rv.size < minimum:
        raise ValueError(
            f"train must be a 1-D realized-variance series with at least {minimum} observations"
        )
    if not np.isfinite(rv).all() or (rv <= 0.0).any():
        raise ValueError("realized-variance series must be finite and strictly positive")
    return rv


@dataclass(frozen=True, eq=False)
class FittedPatchTST:
    """A PatchTST trained on one window; torch-free and picklable.

    The forecast for every horizon ``1..max_horizon`` was computed at fit
    time from the window's last ``lookback`` points, so ``predict`` is pure
    arithmetic. No ``update``: see the module docstring.

    ``eq=False``: numpy fields (see ``Origin`` in splitter.py).
    """

    model: PatchTST
    log_forecast: NDArray[np.float64]
    smearing: NDArray[np.float64]
    training: dict[str, Any]

    @property
    def name(self) -> str:
        return self.model.name

    def spec(self) -> dict[str, Any]:
        return {
            **self.model.spec(),
            **self.training,
            "smearing_factor": [float(s) for s in self.smearing],
            "log_forecast": [float(v) for v in self.log_forecast],
        }

    def predict(self, h: int) -> Distribution:
        if h < 1:
            raise ValueError("h must be >= 1")
        if h > self.model.max_horizon:
            raise ValueError(f"h={h} exceeds max_horizon={self.model.max_horizon}")
        mu = float(self.log_forecast[h - 1])
        factor = float(self.smearing[h - 1])
        if not (math.isfinite(mu) and math.isfinite(factor) and factor > 0.0):
            raise ValueError(f"unusable log forecast {mu!r} / smearing factor {factor!r}")
        vhat = math.exp(mu) * factor
        if not (math.isfinite(vhat) and vhat > 0.0):
            raise ValueError(
                f"retransformed variance forecast is not positive and finite: {vhat!r}"
            )
        return Normal(mu=0.0, sigma=math.sqrt(vhat))


@dataclass(frozen=True)
class PatchTST:
    """PatchTST on log realized variance, smearing-retransformed. See the module docstring.

    Parameters (all hashed except ``device``)
    ----------
    lookback, patch_len, stride:
        Input window length and its patching; ``(lookback - patch_len) //
        stride + 1`` patches. Defaults 64 / 16 / 8 -> 7 patches.
    d_model, n_heads, n_layers, d_ff, dropout:
        Encoder size. Defaults 32 / 4 / 2 / 64 / 0.1 (~20k parameters).
    max_horizon:
        Direct outputs; ``predict(h)`` needs ``h <= max_horizon``.
    max_epochs, patience, val_fraction, batch_size, lr, weight_decay:
        The training budget and early-stop rule (module docstring).
    seed:
        Weight init, batch order and dropout.
    device:
        ``"auto"`` (CUDA if available), ``"cuda"``/``"cuda:0"``, ``"cpu"``.
    """

    lookback: int = 64
    patch_len: int = 16
    stride: int = 8
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.1
    max_horizon: int = 1
    max_epochs: int = 100
    patience: int = 10
    val_fraction: float = 0.2
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if not (1 <= self.patch_len <= self.lookback) or self.stride < 1:
            raise ValueError("need 1 <= patch_len <= lookback and stride >= 1")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.d_model, self.n_heads, self.n_layers, self.d_ff, self.max_horizon) < 1:
            raise ValueError("d_model, n_heads, n_layers, d_ff and max_horizon must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if min(self.max_epochs, self.patience, self.batch_size) < 1:
            raise ValueError("max_epochs, patience and batch_size must be >= 1")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie strictly inside (0, 1)")
        if not (self.lr > 0.0 and self.weight_decay >= 0.0):
            raise ValueError("lr must be > 0 and weight_decay >= 0")

    @property
    def name(self) -> str:
        return "patchtst"

    @property
    def min_train(self) -> int:
        """Smallest window that yields two training windows plus one validation window."""
        return self.lookback + self.max_horizon + 2

    def spec(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "input": "log_realized_variance",
            "normalization": "instance",
            "lookback": self.lookback,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "max_horizon": self.max_horizon,
            "loss": "mse",
            "optimizer": "adam",
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping": "min val mse, restore best",
            "patience": self.patience,
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "retransform": "smearing",
            "torch": _torch_version(),
        }

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedPatchTST:
        rv = _validated_rv(train, minimum=self.min_train)
        from volbench.models._patchtst_net import train_and_forecast

        result = train_and_forecast(
            np.log(rv),
            lookback=self.lookback,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            horizon=self.max_horizon,
            max_epochs=self.max_epochs,
            patience=self.patience,
            val_fraction=self.val_fraction,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            seed=self.seed,
            device=self.device,
        )
        return FittedPatchTST(
            model=self,
            log_forecast=np.asarray(result.log_forecast, dtype=np.float64),
            smearing=np.asarray(result.smearing, dtype=np.float64),
            training={
                "n_train_windows": result.n_train_windows,
                "n_val_windows": result.n_val_windows,
                "epochs_run": result.epochs_run,
                "best_epoch": result.best_epoch,
                "best_val_mse": result.best_val_mse,
                "final_train_mse": result.final_train_mse,
                "stopped_early": result.stopped_early,
            },
        )
