"""`import volbench` must not need any optional backend (Phase-2 integration).

The classical stream kept `models/sf.py` and `models/lgbm.py` out of
`volbench.models` because their backends were imported at module top; the
tsfm stream re-exported its adapters because theirs are imported lazily.
Integration settled on one rule — every adapter is re-exported, every
optional backend is imported inside `fit` — and this is the test that keeps
it true. It runs the import in a subprocess with a meta-path finder that
refuses the backends' top-level packages (the same ``ModuleNotFoundError`` an
uninstalled package raises), so it holds on a machine that has every extra
installed. ``sys.modules[name] = None`` would be the usual trick, but scipy's
array-API shim looks torch up in ``sys.modules`` and trips over the ``None``.
"""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import pytest

from volbench.models import AutoARIMARV, AutoETSRV, LightGBMRV, PatchTST

OPTIONAL_BACKENDS = ("statsforecast", "lightgbm", "torch", "chronos", "timesfm", "uni2ts", "nixtla")

_IMPORT_WITHOUT_BACKENDS = f"""
import sys
from importlib.abc import MetaPathFinder

BLOCKED = set({OPTIONAL_BACKENDS!r})

class _Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}} (blocked by test)", name=name)
        return None

sys.meta_path.insert(0, _Block())
import volbench
import volbench.models
assert not (BLOCKED & set(sys.modules)), sorted(BLOCKED & set(sys.modules))
print(sorted(volbench.models.__all__))
"""


def test_import_volbench_needs_no_optional_backend() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_BACKENDS],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("AutoETSRV", "AutoARIMARV", "LightGBMRV", "PatchTST", "Chronos"):
        assert name in proc.stdout


@pytest.mark.parametrize(
    ("model", "backend"),
    [
        (AutoETSRV(), "statsforecast"),
        (AutoARIMARV(), "statsforecast"),
        (LightGBMRV(), "lightgbm"),
        (PatchTST(device="cpu"), "torch"),
    ],
    ids=["autoets", "autoarima", "lightgbm", "patchtst"],
)
def test_constructing_and_describing_an_adapter_never_touches_its_backend(
    model: AutoETSRV | AutoARIMARV | LightGBMRV | PatchTST, backend: str
) -> None:
    assert isinstance(model.spec(), dict)
    module: ModuleType = sys.modules[type(model).__module__]
    # The backend is not a module-level name: it is imported inside `fit`.
    assert backend not in vars(module), f"{module.__name__} imports {backend} at module level"
