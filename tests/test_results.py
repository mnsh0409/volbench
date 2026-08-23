"""Config hashing and the results store.

The hash is the project's reproducibility primitive: `make reproduce` and
every cache decision rest on "same hash iff same experiment". These tests are
where that claim is nailed down in both directions — stable when nothing
changed, different when anything did.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from volbench.results import (
    KEY_COLUMNS,
    REQUIRED_COLUMNS,
    ResultsStore,
    array_digest,
    build_config,
    canonical_repr,
    config_hash,
    normalize_frame,
    package_version,
)
from volbench.splitter import RollingOriginSplitter

SPLITTER = RollingOriginSplitter(window=10, horizon=1, step=1, refit_every=2)


def base_config() -> dict[str, Any]:
    return build_config(
        model_name="ewma",
        model_spec={"lambda": 0.94, "kind": "ewma"},
        data_spec={"asset": "SPX", "n": 100, "series_sha256": "ab" * 32},
        splitter=SPLITTER,
        seed=7,
        scoring={"levels": [0.01, 0.025, 0.05]},
        version="0.0.1",
    )


# --------------------------------------------------------------------------
# canonical serialization
# --------------------------------------------------------------------------


def test_hash_is_insensitive_to_key_insertion_order() -> None:
    forward = {"a": 1, "b": {"x": 1.0, "y": [1, 2]}, "c": "z"}
    backward = {"c": "z", "b": {"y": [1, 2], "x": 1.0}, "a": 1}
    assert config_hash(forward) == config_hash(backward)
    assert canonical_repr(forward) == canonical_repr(backward)


def test_hash_is_stable_across_processes_and_hash_seeds() -> None:
    """Guards the classic bug: a hash that depends on set/dict iteration order.

    PYTHONHASHSEED randomizes that order per process, so two child processes
    with different seeds would disagree if anything unsorted crept in.
    """
    script = (
        "from volbench.results import config_hash;"
        "print(config_hash({'s': {'b', 'a', 'c'}, 'd': {'k': [1, 2.5, True, None]},"
        " 'f': 0.1, 'g': 'x'}))"
    )
    digests = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1
    assert digests.pop() == config_hash(
        {"s": {"b", "a", "c"}, "d": {"k": [1, 2.5, True, None]}, "f": 0.1, "g": "x"}
    )


def test_hash_is_a_sha256_hex_digest() -> None:
    digest = config_hash(base_config())
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "garch11"),
        ("model_spec", {"lambda": 0.97, "kind": "ewma"}),
        ("data_spec", {"asset": "NDX", "n": 100, "series_sha256": "ab" * 32}),
        ("splitter", RollingOriginSplitter(window=11, horizon=1, step=1, refit_every=2)),
        ("seed", 8),
        ("scoring", {"levels": [0.01, 0.05]}),
        ("version", "0.0.2"),
    ],
)
def test_hash_changes_when_any_component_changes(field: str, value: Any) -> None:
    kwargs: dict[str, Any] = {
        "model_name": "ewma",
        "model_spec": {"lambda": 0.94, "kind": "ewma"},
        "data_spec": {"asset": "SPX", "n": 100, "series_sha256": "ab" * 32},
        "splitter": SPLITTER,
        "seed": 7,
        "scoring": {"levels": [0.01, 0.025, 0.05]},
        "version": "0.0.1",
    }
    assert config_hash(build_config(**kwargs)) == config_hash(base_config())
    kwargs[field] = value
    assert config_hash(build_config(**kwargs)) != config_hash(base_config())


@pytest.mark.parametrize(
    ("splitter", "other"),
    [
        (SPLITTER, RollingOriginSplitter(window=10, horizon=2, step=1, refit_every=2)),
        (SPLITTER, RollingOriginSplitter(window=10, horizon=1, step=2, refit_every=2)),
        (SPLITTER, RollingOriginSplitter(window=10, horizon=1, step=1, refit_every=3)),
    ],
)
def test_every_splitter_parameter_reaches_the_hash(
    splitter: RollingOriginSplitter, other: RollingOriginSplitter
) -> None:
    """A refit schedule that did not change the hash would let a cached run
    serve a different protocol — the cache would silently rewrite the paper."""
    common: dict[str, Any] = {
        "model_name": "ewma",
        "model_spec": {},
        "data_spec": {},
        "seed": 1,
        "version": "0.0.1",
    }
    assert config_hash(build_config(splitter=splitter, **common)) != config_hash(
        build_config(splitter=other, **common)
    )


def test_types_are_tagged_so_look_alike_values_do_not_collide() -> None:
    digests = {
        config_hash({"a": 1}),
        config_hash({"a": 1.0}),
        config_hash({"a": "1"}),
        config_hash({"a": True}),
        config_hash({"a": None}),
        config_hash({"a": [1]}),
    }
    assert len(digests) == 6


def test_float_formatting_round_trips_exactly() -> None:
    assert config_hash({"x": 0.1 + 0.2}) != config_hash({"x": 0.3})
    assert config_hash({"x": 1e-17}) != config_hash({"x": 0.0})
    # Values that compare equal must hash equal.
    assert config_hash({"x": -0.0}) == config_hash({"x": 0.0})
    assert config_hash({"x": 1.0}) == config_hash({"x": float(1)})


def test_non_finite_floats_are_representable() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        assert len(config_hash({"x": value})) == 64
    assert config_hash({"x": float("nan")}) == config_hash({"x": float("nan")})
    assert config_hash({"x": float("inf")}) != config_hash({"x": float("-inf")})


def test_numpy_scalars_hash_like_their_python_equivalents() -> None:
    assert config_hash({"a": np.int64(3)}) == config_hash({"a": 3})
    assert config_hash({"a": np.float64(0.5)}) == config_hash({"a": 0.5})
    assert config_hash({"a": np.bool_(True)}) == config_hash({"a": True})
    assert config_hash({"a": np.array([1.0, 2.0])}) == config_hash({"a": [1.0, 2.0]})


def test_dates_are_representable() -> None:
    assert config_hash({"freeze": dt.date(2026, 8, 23)}) != config_hash(
        {"freeze": dt.date(2026, 8, 24)}
    )


def test_list_order_matters_but_set_order_does_not() -> None:
    assert config_hash({"a": [1, 2]}) != config_hash({"a": [2, 1]})
    assert config_hash({"a": {1, 2}}) == config_hash({"a": {2, 1}})


def test_paths_are_rejected_because_they_differ_across_machines() -> None:
    """D-011 runs the same cells on the dev box and on the cluster; a config
    carrying an absolute path could never produce a matching hash there."""
    with pytest.raises(TypeError, match="paths must not appear"):
        config_hash({"cache": Path("/home/martin/data")})


def test_unhashable_types_fail_loudly_rather_than_hashing_a_repr() -> None:
    with pytest.raises(TypeError, match="cannot canonicalize"):
        config_hash({"model": object()})
    with pytest.raises(TypeError, match="raw bytes"):
        config_hash({"payload": b"\x00\x01"})
    with pytest.raises(TypeError, match="keys must be str"):
        config_hash({"outer": {1: "a"}})
    with pytest.raises(TypeError, match="must be a mapping"):
        config_hash([1, 2])  # type: ignore[arg-type]


def test_package_version_is_the_installed_one() -> None:
    """The version in a config hash must be the real installed one.

    Compared against ``volbench.__version__`` rather than a literal: pinning
    the literal here meant every release bumped an unrelated test (it did, at
    the M1 version bump). What actually matters is that the two agree — a
    config hash carrying a *different* version from the package that produced
    the numbers is a provenance bug — and that neither is the not-installed
    fallback.
    """
    import volbench

    assert package_version() == volbench.__version__
    assert package_version() != "0+unknown", "volbench is not installed in this environment"
    assert base_config()["package_version"] == "0.0.1"  # explicit override, not the installed one


def test_array_digest_tracks_content_not_identity() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert array_digest(a) == array_digest(a.copy())
    assert array_digest(a) != array_digest(np.array([1.0, 2.0, 3.5]))
    assert array_digest(a) != array_digest(np.array([1.0, 2.0, 3.0, 4.0]))
    assert array_digest(a) == array_digest(np.array([1, 2, 3], dtype=np.int64))


# --------------------------------------------------------------------------
# frames and store
# --------------------------------------------------------------------------


def make_frame(digest: str, n: int = 4, crps: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": pd.array(["SPX"] * n, dtype="str"),
            "crps": np.full(n, crps),
            "config_hash": pd.array([digest] * n, dtype="str"),
            "horizon": np.ones(n, dtype=np.int64),
            "origin_index": np.arange(n, dtype=np.int64),
            "forecast_mean": np.zeros(n),
            "forecast_var": np.full(n, 1e-4),
            "realized_return": np.linspace(-0.01, 0.01, n),
            "proxy_name": pd.array(["parkinson"] * n, dtype="str"),
            "proxy_var": np.full(n, 1.2e-4),
        }
    )


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def test_normalize_frame_fixes_column_and_row_order() -> None:
    frame = make_frame(DIGEST_A).iloc[::-1]
    out = normalize_frame(frame)
    assert list(out.columns[: len(KEY_COLUMNS)]) == list(KEY_COLUMNS)
    assert list(out.columns[len(KEY_COLUMNS) :]) == sorted(out.columns[len(KEY_COLUMNS) :])
    assert out["origin_index"].tolist() == [0, 1, 2, 3]
    assert out.index.tolist() == [0, 1, 2, 3]


def test_normalize_frame_rejects_frames_missing_provenance() -> None:
    frame = make_frame(DIGEST_A).drop(columns=["proxy_var"])
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_frame(frame)


def test_required_columns_include_the_key() -> None:
    assert set(KEY_COLUMNS) <= set(REQUIRED_COLUMNS)


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    frame = normalize_frame(make_frame(DIGEST_A))
    assert store.write(frame) == [DIGEST_A]
    assert store.has(DIGEST_A)
    pd.testing.assert_frame_equal(store.read(DIGEST_A), frame)


def test_rerunning_an_identical_config_does_not_duplicate_rows(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    frame = make_frame(DIGEST_A)
    store.write(frame)
    assert store.write(frame) == []  # second write is a no-op
    assert len(store.read_all()) == len(frame)
    assert store.config_hashes() == [DIGEST_A]


def test_a_stored_config_is_never_silently_replaced(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    store.write(make_frame(DIGEST_A, crps=0.5))
    store.write(make_frame(DIGEST_A, crps=99.0))
    assert store.read(DIGEST_A)["crps"].tolist() == [0.5] * 4


def test_overwrite_replaces_deliberately(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    store.write(make_frame(DIGEST_A, crps=0.5))
    assert store.write(make_frame(DIGEST_A, crps=99.0), overwrite=True) == [DIGEST_A]
    assert store.read(DIGEST_A)["crps"].tolist() == [99.0] * 4


def test_one_write_can_carry_several_cells(tmp_path: Path) -> None:
    """Cells merge by config_hash — a grid result lands as several fragments."""
    store = ResultsStore(tmp_path / "results")
    combined = pd.concat([make_frame(DIGEST_A), make_frame(DIGEST_B)], ignore_index=True)
    assert store.write(combined) == [DIGEST_A, DIGEST_B]
    assert store.config_hashes() == [DIGEST_A, DIGEST_B]
    assert len(store.read_all()) == 8
    assert store.read(DIGEST_A)["config_hash"].unique().tolist() == [DIGEST_A]


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    frame = make_frame(DIGEST_A)
    doubled = pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate result keys"):
        store.write(doubled)


def test_missing_key_values_are_rejected(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    frame = make_frame(DIGEST_A)
    frame.loc[0, "asset"] = None
    with pytest.raises(ValueError, match="must not contain missing values"):
        store.write(frame)


def test_config_sidecar_records_how_rows_were_produced(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    config = base_config()
    store.write(make_frame(DIGEST_A), config=config)
    stored = store.read_config(DIGEST_A)
    assert stored["model"]["name"] == "ewma"
    assert stored["seed"] == 7
    assert stored["splitter"]["refit_every"] == 2
    # The sidecar is documentation; the hash is computed from the config itself.
    assert json.loads(json.dumps(stored)) == stored


def test_reading_an_unknown_hash_raises(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    assert not store.has(DIGEST_A)
    with pytest.raises(KeyError):
        store.read(DIGEST_A)
    with pytest.raises(KeyError):
        store.read_config(DIGEST_A)


def test_empty_store_reads_as_an_empty_frame(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "results")
    empty = store.read_all()
    assert len(empty) == 0
    assert set(REQUIRED_COLUMNS) <= set(empty.columns)


def test_hash_shaped_strings_only(tmp_path: Path) -> None:
    """The hash becomes a filename, so anything else is a path-traversal bug."""
    store = ResultsStore(tmp_path / "results")
    for bad in ("../evil", "A" * 64, "abc", ""):
        with pytest.raises(ValueError, match="not a valid config hash"):
            store.fragment_path(bad)


def test_partial_writes_are_never_visible(tmp_path: Path) -> None:
    """Fragments land via os.replace, so no temp file can be mistaken for
    results by a later read_all()."""
    store = ResultsStore(tmp_path / "results")
    store.write(make_frame(DIGEST_A), config=base_config())
    leftovers = [p.name for p in store.root.iterdir() if ".tmp-" in p.name]
    assert leftovers == []
    assert sorted(p.name for p in store.root.iterdir()) == [
        f"{DIGEST_A}.json",
        f"{DIGEST_A}.parquet",
    ]
