"""Config hashing and the append-only results store.

Two jobs, both in service of CLAUDE.md rule 3 (determinism):

1. :func:`config_hash` turns the full description of a run — model spec, data
   *contents*, splitter parameters, scoring parameters, seed, package version
   — into one stable SHA-256. Two runs share a hash if and only if they are
   the same experiment.
2. :class:`ResultsStore` persists scored rows keyed by that hash, so a run
   that has already been done is never redone, and so results computed on
   different execution backends merge without coordination (D-011).

Hashing the data *contents*, not just a data label, is a leakage control, not
bookkeeping: it makes it impossible for a cached artifact computed from one
series to be served for a different (e.g. later, longer, revised) one. See
``.claude/skills/leakage-check`` item 9.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # no stubs; pyproject is another stream's file
from numpy.typing import NDArray

from volbench.splitter import RollingOriginSplitter

__all__ = [
    "KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "ResultsStore",
    "array_digest",
    "build_config",
    "canonical_repr",
    "config_hash",
    "normalize_frame",
    "package_version",
]

#: Uniqueness key for a result row. One row per cell per forecast target.
KEY_COLUMNS: Final = ("config_hash", "asset", "origin_index", "horizon")

#: Provenance every row must carry, whatever scores a given run produced.
REQUIRED_COLUMNS: Final = (
    *KEY_COLUMNS,
    "forecast_mean",
    "forecast_var",
    "realized_return",
    "proxy_name",
    "proxy_var",
)

_HASH_RE: Final = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------
# canonical serialization
# --------------------------------------------------------------------------


def package_version() -> str:
    """Installed volbench version, or ``"0+unknown"`` outside an install.

    Read through :mod:`importlib.metadata` rather than ``volbench.__version__``
    so that this module never imports the package root — the root will import
    *this* module once the Phase 1 streams are wired together.
    """
    try:
        return importlib.metadata.version("volbench")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - dev fallback
        return "0+unknown"


def _fmt_float(x: float) -> str:
    """Fixed float formatting: exact round-trip, one spelling per value."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0.0 else "-inf"
    if x == 0.0:
        return "0"  # collapse -0.0 and 0.0, which compare equal
    return f"{x:.17g}"  # 17 significant digits round-trips float64 exactly


def _canon(obj: Any) -> str:
    """Canonical string for one value.

    Values are type-tagged, so ``1``, ``1.0`` and ``"1"`` hash differently: a
    spec whose type changed is a spec that changed, and silently colliding
    them would let two different experiments share a cache entry.
    """
    if obj is None:
        return "null"
    # bool first: bool is a subclass of int.
    if isinstance(obj, bool | np.bool_):
        return "true" if obj else "false"
    if isinstance(obj, int | np.integer):
        return f"i:{int(obj)}"
    if isinstance(obj, float | np.floating):
        return f"f:{_fmt_float(float(obj))}"
    if isinstance(obj, str):
        return "s:" + json.dumps(obj, ensure_ascii=True)
    if isinstance(obj, bytes | bytearray):
        # Caught before the Sequence branch, which would silently canonicalize
        # bytes as a list of ints.
        raise TypeError("raw bytes must not appear in a config; use a hex digest instead")
    if isinstance(obj, dt.datetime | dt.date):
        return "t:" + obj.isoformat()
    if isinstance(obj, Mapping):
        keys = list(obj.keys())
        if not all(isinstance(k, str) for k in keys):
            raise TypeError(f"config keys must be str, got {[type(k).__name__ for k in keys]}")
        inner = ",".join(f"{_canon(k)}={_canon(obj[k])}" for k in sorted(keys))
        return "{" + inner + "}"
    if isinstance(obj, frozenset | set):
        return "<" + ",".join(sorted(_canon(v) for v in obj)) + ">"
    if isinstance(obj, np.ndarray):
        return "[" + ",".join(_canon(v) for v in obj.tolist()) + "]"
    if isinstance(obj, os.PathLike):
        # Filesystem paths differ between the dev box and the cluster, so a
        # path in a config would break the backend-invariance claim (D-011).
        raise TypeError(
            "filesystem paths must not appear in a config: they differ across machines. "
            "Use a stable logical identifier (asset id, source tag, checksum) instead."
        )
    if isinstance(obj, Sequence):
        return "[" + ",".join(_canon(v) for v in obj) + "]"
    raise TypeError(
        f"cannot canonicalize {type(obj).__name__}; config values must be JSON-like "
        "(str, int, float, bool, None, date, mapping, sequence, set)"
    )


def canonical_repr(spec: Mapping[str, Any]) -> str:
    """Canonical serialization of ``spec`` — the exact bytes that get hashed.

    Public so that a hash mismatch is diagnosable by diffing two of these
    rather than staring at two hex digests.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"spec must be a mapping, got {type(spec).__name__}")
    return _canon(spec)


def config_hash(spec: Mapping[str, Any]) -> str:
    """Stable SHA-256 over ``spec``, insensitive to key insertion order."""
    return hashlib.sha256(canonical_repr(spec).encode("utf-8")).hexdigest()


def array_digest(values: NDArray[np.float64]) -> str:
    """SHA-256 of an array's float64 bytes — identifies data by content."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def build_config(
    *,
    model_name: str,
    model_spec: Mapping[str, Any],
    data_spec: Mapping[str, Any],
    splitter: RollingOriginSplitter,
    seed: int,
    scoring: Mapping[str, Any] | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Assemble the config dict that :func:`config_hash` consumes.

    ``splitter`` is typed as :class:`~volbench.splitter.RollingOriginSplitter`
    on purpose: it is the only sanctioned source of train/test indices
    (CLAUDE.md rule 1), so a hand-rolled splitter cannot even be described
    here, let alone cached.
    """
    return {
        "model": {"name": model_name, "spec": dict(model_spec)},
        "data": dict(data_spec),
        "splitter": {"class": type(splitter).__name__, **asdict(splitter)},
        "scoring": dict(scoring or {}),
        "seed": int(seed),
        "package_version": version if version is not None else package_version(),
    }


# --------------------------------------------------------------------------
# frame normalization
# --------------------------------------------------------------------------


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Put a results frame into canonical form: fixed column and row order.

    Backend invariance depends on this. Origins may be scored in any order by
    any executor; normalizing before comparison or persistence means the bytes
    on disk depend on the *content* of a run and nothing else.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"results frame is missing required columns: {missing}")
    rest = sorted(c for c in frame.columns if c not in KEY_COLUMNS)
    ordered = frame[[*KEY_COLUMNS, *rest]]
    return ordered.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


class ResultsStore:
    """Append-only parquet store, one fragment per ``config_hash``.

    Layout::

        <root>/<config_hash>.parquet   scored rows for that cell
        <root>/<config_hash>.json      the config those rows came from

    A fragment is written once and never mutated, which gives three
    properties this project needs:

    - **Idempotence.** Re-running an identical config is a no-op, so a
      restarted grid cannot double-count rows.
    - **Cache short-circuit.** :meth:`has` is a file-existence check, so
      skipping completed work costs no I/O over the results themselves.
    - **Lock-free merging.** Distinct cells write distinct paths, so local
      processes and Slurm array tasks merge by simply landing in the same
      directory (D-011).

    Writes go through a temporary file and :func:`os.replace`, so a killed job
    leaves either the old fragment or the new one — never a half-written one
    that a later ``read_all`` would silently treat as real results.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"ResultsStore({str(self.root)!r})"

    # --- addressing ------------------------------------------------------

    def fragment_path(self, config_hash_: str) -> Path:
        """Path of the parquet fragment for ``config_hash_``."""
        return self.root / f"{self._validated(config_hash_)}.parquet"

    def config_path(self, config_hash_: str) -> Path:
        """Path of the JSON config sidecar for ``config_hash_``."""
        return self.root / f"{self._validated(config_hash_)}.json"

    @staticmethod
    def _validated(config_hash_: str) -> str:
        # Also blocks path traversal: this string becomes a filename.
        if not isinstance(config_hash_, str) or not _HASH_RE.match(config_hash_):
            raise ValueError(f"not a valid config hash: {config_hash_!r}")
        return config_hash_

    # --- queries ---------------------------------------------------------

    def has(self, config_hash_: str) -> bool:
        """Whether results for ``config_hash_`` are already stored."""
        return self.fragment_path(config_hash_).is_file()

    def config_hashes(self) -> list[str]:
        """Every stored config hash, sorted."""
        return sorted(p.stem for p in self.root.glob("*.parquet") if _HASH_RE.match(p.stem))

    # --- reads -----------------------------------------------------------

    def read(self, config_hash_: str) -> pd.DataFrame:
        """Read one cell's rows. Raises :class:`KeyError` if not stored."""
        path = self.fragment_path(config_hash_)
        if not path.is_file():
            raise KeyError(f"no results stored for config hash {config_hash_}")
        frame: pd.DataFrame = pd.read_parquet(path)
        return frame

    def read_config(self, config_hash_: str) -> dict[str, Any]:
        """Read the config a stored cell was produced from."""
        path = self.config_path(config_hash_)
        if not path.is_file():
            raise KeyError(f"no config stored for config hash {config_hash_}")
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def read_all(self) -> pd.DataFrame:
        """Every stored row, concatenated in config-hash order."""
        hashes = self.config_hashes()
        if not hashes:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in REQUIRED_COLUMNS})
        return pd.concat([self.read(h) for h in hashes], ignore_index=True)

    # --- writes ----------------------------------------------------------

    def write(
        self,
        frame: pd.DataFrame,
        *,
        config: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> list[str]:
        """Persist ``frame``, one fragment per ``config_hash`` it contains.

        Returns the hashes actually written. Hashes already present are
        skipped unless ``overwrite`` is set — that skip is what makes a
        re-run idempotent rather than duplicating rows.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"frame must be a DataFrame, got {type(frame).__name__}")
        frame = normalize_frame(frame)
        if frame[list(KEY_COLUMNS)].isna().to_numpy().any():
            raise ValueError(f"key columns {KEY_COLUMNS} must not contain missing values")
        duplicated = frame.duplicated(subset=list(KEY_COLUMNS))
        if bool(duplicated.any()):
            dupes = frame.loc[duplicated, list(KEY_COLUMNS)].to_dict("records")
            raise ValueError(f"duplicate result keys in frame: {dupes[:5]}")

        written: list[str] = []
        for hash_value, group in frame.groupby("config_hash", sort=True):
            key = self._validated(str(hash_value))
            if self.has(key) and not overwrite:
                continue
            self._write_fragment(key, group.reset_index(drop=True))
            if config is not None:
                self._write_config(key, config)
            written.append(key)
        return written

    def _write_fragment(self, key: str, group: pd.DataFrame) -> None:
        path = self.fragment_path(key)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        group.to_parquet(tmp, index=False, compression="snappy", engine="pyarrow")
        os.replace(tmp, path)

    def _write_config(self, key: str, config: Mapping[str, Any]) -> None:
        path = self.config_path(key)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(
            json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)


def _json_safe(obj: Any) -> Any:
    """Coerce a config to plain JSON types for the sidecar.

    The sidecar is documentation; :func:`canonical_repr` remains the thing
    that is hashed, so lossy coercions here cannot change any hash.
    """
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dt.datetime | dt.date):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(obj.items())}
    if isinstance(obj, frozenset | set):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, str | int | float) or obj is None:
        return obj
    if isinstance(obj, Iterable):
        return [_json_safe(v) for v in obj]
    return str(obj)
