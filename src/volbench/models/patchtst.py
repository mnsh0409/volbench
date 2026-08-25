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
Across devices: with ``dropout=0`` CPU and CUDA fits agree to float rounding
(measured ~1e-8 relative); with dropout on, each device draws its masks from
its own RNG stream, so a CPU fit and a CUDA fit of the same seed are two
training realisations (~1.6% apart on a 300-point window). Results therefore
reproduce *per device class*.

**The device class is hashed (D-031).** Up to v0.4.0 ``device`` was outside
``spec()`` altogether, on the reading that it is *where* the computation runs
rather than *what* is computed. The line above is why that was wrong for this
model: with dropout on, the device class picks the RNG stream and therefore
the training realization, so two numerically different fragments could share
one ``config_hash`` and the store could serve either for the other. Since
v0.5.0 ``spec()`` carries ``device_class`` — ``"cuda"`` or ``"cpu"``, the
torch device *type*, not the ordinal — so a CPU fragment and a GPU fragment
of the same configuration are two different cells and neither can be served
for the other. The ordinal stays out: ``"cuda:0"`` and ``"cuda:1"`` are the
same class and hash identically, which is what lets a multi-GPU grid place a
cell on whichever card is free. Two *different GPU models* are not
distinguished either; that is the same qualification D-026 puts on the numpy
kernel family, and the paper's runs state their hardware.

``device="auto"`` resolves against the machine (CUDA if available), so its
``spec()`` — and hence the cell's identity — depends on where it is
evaluated. That is the honest answer for a setting whose meaning is "whatever
this box has": pin ``device="cuda"`` or ``"cpu"`` to fix the identity in
advance. Resolving ``"auto"`` needs torch importable; an explicit device
never does, so describing a pinned configuration still costs no backend.

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
from volbench.models._rv import validated_rv, variance_from_log

__all__ = ["FittedPatchTST", "PatchTST", "resolve_device_class"]

# cuBLAS reads this when it creates its handles, which happens on the first
# CUDA op in the process; deterministic mode refuses cuBLAS calls otherwise.
# Set at import so any volbench process is covered before it touches CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _torch_version() -> str:
    try:
        return version("torch")
    except PackageNotFoundError:
        return "not-installed"


def resolve_device_class(device: str) -> str:
    """The torch device *type* ``device`` names on this machine (D-031).

    ``"cuda:1"`` -> ``"cuda"``: the ordinal picks a card, not a training
    realization, so it stays out of the config hash. ``"auto"`` is resolved
    against the machine, which is the only honest reading of it — and the one
    case that needs torch importable, because "whatever this box has" is not
    answerable without asking.
    """
    if device == "auto":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "device='auto' cannot be resolved without torch installed, and the device "
                "class is part of this model's config hash (D-031). Pin device='cpu' or "
                "device='cuda' to describe the configuration without a backend."
            ) from exc
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device.split(":", 1)[0]


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
        if not (math.isfinite(factor) and factor > 0.0):
            raise ValueError(f"unusable smearing factor {factor!r}")
        # Same retransformation as models/sf.py and models/lgbm.py (Duan
        # smearing, `volbench.models._rv`); the factor itself is per horizon.
        return Normal(mu=0.0, sigma=math.sqrt(variance_from_log(mu, factor)))


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
        The string itself is not hashed; the *class* it resolves to is
        (``device_class``, D-031 — see the module docstring).
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
    def device_class(self) -> str:
        """``"cuda"`` or ``"cpu"`` — the hashed half of ``device`` (D-031)."""
        return resolve_device_class(self.device)

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
            # D-031: the device class picks the dropout RNG stream and hence
            # the training realization, so it identifies the cell. The ordinal
            # does not and is deliberately absent.
            "device_class": self.device_class,
        }

    def fit(self, train: NDArray[np.float64], **ctx: Any) -> FittedPatchTST:
        rv = validated_rv(train, minimum=self.min_train)
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
