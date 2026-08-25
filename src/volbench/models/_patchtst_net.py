"""Torch side of :mod:`volbench.models.patchtst`: the network and the bounded training loop.

Imported lazily by ``PatchTST.fit`` so that ``volbench.models`` never needs
torch at import time. Everything that touches the model *contract* — what
the network is fed, what comes back, how log forecasts become variances —
is decided in ``patchtst.py``; this module only knows tensors.

Determinism, which is a gate for this baseline:

- ``torch.use_deterministic_algorithms(True)`` is switched on for the
  duration of a fit (and restored afterwards — it is process-global and
  the TSFM pipelines running in the same process must not inherit it).
- Attention is written out as plain ``matmul`` + ``softmax``: the fused
  scaled-dot-product kernels are not all deterministic, and a model this
  small gains nothing from them.
- No op with a non-deterministic CUDA implementation is used (no
  ``index_add``/``scatter`` in the backward path, no ``cudnn`` convolutions).
  cuBLAS needs ``CUBLAS_WORKSPACE_CONFIG`` set before its handles exist;
  ``patchtst.py`` sets it at import.
- Mini-batch order comes from a CPU ``torch.Generator`` seeded with the
  model's seed, and weights are initialised on the CPU before ``.to(device)``,
  so init and batch order are device-independent. Dropout is not: it draws
  from the *device's* RNG stream (seeded by the same seed), and the CPU and
  CUDA streams differ. Same window + same seed => identical weights and
  forecast on one device; across devices, ``dropout=0`` fits agree to float
  rounding and ``dropout>0`` fits are different realisations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from volbench.models._rv import smearing_factor

__all__ = ["PatchTSTNet", "TrainingResult", "train_and_forecast"]

#: Floor on an input window's standard deviation before instance normalization.
_STD_FLOOR = 1e-5


@dataclass(frozen=True, eq=False)
class TrainingResult:
    """What one bounded fit produced. ``eq=False``: numpy fields."""

    log_forecast: NDArray[np.float64]  # (H,) log-RV forecast for steps 1..H past the window
    smearing: NDArray[np.float64]  # (H,) Duan smearing factor per horizon, from training residuals
    epochs_run: int
    best_epoch: int
    best_val_mse: float  # in normalized log space, the quantity early stopping watched
    final_train_mse: float  # same units, at the restored (best) weights
    n_train_windows: int
    n_val_windows: int
    stopped_early: bool


class _Attention(nn.Module):  # type: ignore[misc]
    """Multi-head self-attention with explicit matmuls (see module docstring)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Any) -> Any:
        batch, n_tokens, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, n_tokens, self.n_heads, self.d_head)
        q = q.reshape(shape).transpose(1, 2)  # (B, heads, tokens, d_head)
        k = k.reshape(shape).transpose(1, 2)
        v = v.reshape(shape).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = self.drop(torch.softmax(scores, dim=-1))
        mixed = torch.matmul(weights, v).transpose(1, 2).reshape(batch, n_tokens, d_model)
        return self.out(mixed)


class _EncoderLayer(nn.Module):  # type: ignore[misc]
    """Pre-norm transformer encoder layer (LayerNorm, GELU feed-forward)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _Attention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Any) -> Any:
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.ff(self.norm2(x)))


class PatchTSTNet(nn.Module):  # type: ignore[misc]
    """Channel-independent PatchTST for one series: patch, embed, encode, flatten, project.

    Input ``(batch, lookback)`` already instance-normalized; output
    ``(batch, horizon)`` in the same normalized units.
    """

    def __init__(
        self,
        *,
        lookback: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        horizon: int,
    ) -> None:
        super().__init__()
        if patch_len > lookback or stride < 1:
            raise ValueError("need patch_len <= lookback and stride >= 1")
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.empty(self.n_patches, d_model))
        nn.init.uniform_(self.pos, -0.02, 0.02)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, horizon)

    def forward(self, x: Any) -> Any:
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
        z = self.drop(self.embed(patches) + self.pos)
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)
        return self.head(self.drop(z.flatten(1)))


def _resolve_device(device: str) -> Any:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _windows(y: NDArray[np.float64], lookback: int, horizon: int) -> tuple[Any, Any]:
    """Every (input, target) pair the window ``y`` contains — and nothing past it.

    ``x_i = y[i : i+L]``, ``t_i = y[i+L : i+L+H]`` for ``i = 0 .. n-L-H``: the
    last target is ``y[n-1]``, the window's own final observation, so no
    batch can reach beyond the origin the caller's ``y`` ends at.
    """
    n = y.size
    n_windows = n - lookback - horizon + 1
    idx = np.arange(n_windows)[:, None]
    x = torch.as_tensor(y[idx + np.arange(lookback)[None, :]], dtype=torch.float32)
    t = torch.as_tensor(y[idx + lookback + np.arange(horizon)[None, :]], dtype=torch.float32)
    return x, t


def _instance_stats(x: Any) -> tuple[Any, Any]:
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(_STD_FLOOR)
    return mu, sd


def train_and_forecast(
    y: NDArray[np.float64],
    *,
    lookback: int,
    patch_len: int,
    stride: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    horizon: int,
    max_epochs: int,
    patience: int,
    val_fraction: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> TrainingResult:
    """Fit on the log-RV window ``y`` under a bounded budget; return the forecast past its end.

    Budget: at most ``max_epochs`` passes over the training windows, stopped
    once validation MSE has not improved for ``patience`` consecutive epochs;
    the weights with the best validation MSE are restored. Validation is the
    chronologically *last* ``val_fraction`` of the windows — never a random
    subset, so the model is selected on the most recent data it may see.
    """
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        dev = _resolve_device(device)

        x_all, t_all = _windows(y, lookback, horizon)
        n_windows = int(x_all.shape[0])
        n_val = max(1, math.ceil(val_fraction * n_windows))
        n_train = n_windows - n_val
        if n_train < 2:
            raise ValueError(
                f"window of {y.size} points yields {n_windows} training windows; "
                f"need at least 2 for training plus {n_val} for validation"
            )
        mu, sd = _instance_stats(x_all)
        xn = ((x_all - mu) / sd).to(dev)
        tn = ((t_all - mu) / sd).to(dev)
        x_tr, t_tr = xn[:n_train], tn[:n_train]
        x_val, t_val = xn[n_train:], tn[n_train:]

        model = PatchTSTNet(
            lookback=lookback,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            horizon=horizon,
        ).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        order = torch.Generator().manual_seed(seed)

        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_val = math.inf
        best_epoch = 0
        epochs_run = 0
        bad_epochs = 0
        stopped_early = False
        for epoch in range(1, max_epochs + 1):
            model.train()
            perm = torch.randperm(n_train, generator=order).to(dev)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x_tr[idx]), t_tr[idx])
                loss.backward()
                optimizer.step()
            epochs_run = epoch
            model.eval()
            with torch.no_grad():
                val = float(loss_fn(model(x_val), t_val).item())
            if val < best_val:
                best_val, best_epoch, bad_epochs = val, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    stopped_early = True
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_train_mse = float(loss_fn(model(x_tr), t_tr).item())
            # in-sample residuals in log space, per horizon, from the training windows
            pred_tr = model(x_tr).cpu() * sd[:n_train] + mu[:n_train]
            resid = (t_all[:n_train] - pred_tr).to(torch.float64).numpy()
            # Duan's factor per horizon column, through the one shared
            # implementation (models/_rv.py) every log-RV model retransforms with.
            smearing = np.array(
                [smearing_factor(resid[:, k]) for k in range(resid.shape[1])],
                dtype=np.float64,
            )
            # the forecast: the last `lookback` points of the window, and nothing after
            x_last = torch.as_tensor(y[-lookback:], dtype=torch.float32)[None, :]
            mu_last, sd_last = _instance_stats(x_last)
            out = model(((x_last - mu_last) / sd_last).to(dev)).cpu() * sd_last + mu_last
        return TrainingResult(
            log_forecast=out[0].to(torch.float64).numpy(),
            smearing=smearing,
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            best_val_mse=best_val,
            final_train_mse=final_train_mse,
            n_train_windows=n_train,
            n_val_windows=n_val,
            stopped_early=stopped_early,
        )
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev_cudnn
