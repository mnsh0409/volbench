"""Tests for the loss-table and pairwise-complete driver.

The numerics live in :mod:`volbench.analysis` and are tested there. What is
left here is the part this module owns: a structural guarantee that the whole
§2/§3 pipeline cannot reach a model, and renderers that must not silently drop
or transpose a number on its way into a published table.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volbench import analysis
from volbench.benchmarks import loss_tables


def _volbench_imports(module: object) -> set[str]:
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")  # type: ignore[arg-type]
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return {name for name in imported if name.startswith("volbench")}


class TestBoundary:
    """This driver orchestrates the analysis layer; it never fits either.

    ``analysis.py`` is held to the boundary by ``tests/test_analysis.py``. That
    is worth nothing here if the module that *calls* it can reach a model and
    slip a fitted number into the same table, so the driver is held to it too —
    with the store reader allowed, as the analysis layer allows it.
    """

    def test_it_reaches_no_model_and_no_evaluator(self) -> None:
        assert _volbench_imports(loss_tables) <= {
            "volbench",
            "volbench.analysis",
            "volbench.results",
        }

    def test_the_layer_it_leans_on_is_still_boundaried(self) -> None:
        for name in _volbench_imports(analysis):
            assert not name.startswith("volbench.models"), name


def _table() -> pd.DataFrame:
    rows = []
    for model in ("alpha", "beta"):
        for loss, mean in (("crps", 0.5), ("qlike", 1.5)):
            rows.append(
                {
                    "asset": "AAA",
                    "model": model,
                    "loss": loss,
                    "origins": 100,
                    "n": 98 if loss == "qlike" else 100,
                    "n_dropped": 2 if loss == "qlike" else 0,
                    "mean": mean,
                    "bandwidth": 4,
                    "se": 0.125,
                    "se_iid": 0.1,
                }
            )
    return pd.DataFrame(rows)


class TestRendering:
    def test_the_loss_table_carries_both_n_columns_and_the_bandwidth(self) -> None:
        """``n`` differs between QLIKE and the distribution losses inside one
        cell; a table showing one of them would misdescribe the other."""
        rendered = loss_tables.loss_table_markdown(_table(), "AAA")
        assert "100 origins" in rendered
        assert "Bandwidth 4" in rendered
        assert analysis.HAC_BANDWIDTH_RULE in rendered
        assert "| `alpha` | 100 | 98 |" in rendered
        assert "0.5 (0.125)" in rendered

    def test_the_matrices_are_rendered_row_by_row_in_model_order(self) -> None:
        """A transposed matrix is symmetric here and would look right. The
        model order of the header and of each row label must agree, so a reader
        indexing by name gets the entry they asked for."""
        grid = pd.DataFrame(
            {
                "asset": "AAA",
                "model_label": ["alpha"] * 3 + ["beta"] * 3,
                "origin_index": [0, 1, 2] * 2,
                "crps": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            }
        )
        result = analysis.pairwise_complete(grid, "AAA", "crps")
        rendered = loss_tables.pairwise_markdown(result)
        assert "| | `alpha` | `beta` |" in rendered
        assert "| `alpha` | 2 | 2 |" in rendered
        assert "| `beta` | 2 | 3 |" in rendered
        assert "**n used**" in rendered and "**rows dropped**" in rendered

    def test_frame_markdown_keeps_every_row_and_column(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        lines = loss_tables.frame_markdown(frame).splitlines()
        assert lines[0] == "| a | b |"
        assert len(lines) == 4
        assert lines[-1] == "| 2 | y |"


class TestLargestDrops:
    def test_it_reports_the_diagonal_and_the_off_diagonal_separately(self) -> None:
        """``dropped[i, i]`` is what a model loses on its own and is often the
        maximum; a comparison needs the worst *pair*, which is a different
        cell."""
        long = pd.DataFrame(
            [
                {"asset": "AAA", "loss": "crps", "model_a": "a", "model_b": "a",
                 "origins": 10, "n_used": 4, "dropped": 6},
                {"asset": "AAA", "loss": "crps", "model_a": "a", "model_b": "b",
                 "origins": 10, "n_used": 7, "dropped": 3},
                {"asset": "AAA", "loss": "crps", "model_a": "b", "model_b": "b",
                 "origins": 10, "n_used": 10, "dropped": 0},
            ]
        )
        answer = loss_tables.largest_drops(long).iloc[0]
        assert answer["largest_drop"] == 6
        assert answer["pair"] == "a / a"
        assert answer["largest_off_diagonal_drop"] == 3
        assert answer["off_diagonal_pair"] == "a / b"


class TestNoCrossAssetAggregation:
    def test_a_loss_table_is_built_for_one_asset_at_a_time(self) -> None:
        """Equity and crypto score against different targets, so a pooled mean
        would be a mean over two units. ``loss_table`` takes one asset and
        cannot be handed several."""
        grid = pd.DataFrame(
            {
                "asset": ["AAA", "BBB"],
                "model_label": ["m", "m"],
                "origin_index": [0, 0],
                "crps": [1.0, 100.0],
            }
        )
        table = analysis.loss_table(grid, "AAA", losses=["crps"])
        assert set(table["asset"]) == {"AAA"}
        assert table["mean"].iloc[0] == pytest.approx(1.0)
