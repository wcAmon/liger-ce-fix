import re
import torch

import liger_ce_fix


def test_version_and_commit_are_strings():
    assert re.fullmatch(r"\d+\.\d+\.\d+", liger_ce_fix.__version__)
    assert isinstance(liger_ce_fix.__commit__, str) and len(liger_ce_fix.__commit__) >= 7


def test_import_does_not_require_triton(monkeypatch):
    # The package must import on a CPU-only box: Triton is only touched inside the CUDA branches.
    import importlib, sys
    monkeypatch.setitem(sys.modules, "triton", None)
    for name in [m for m in sys.modules if m.startswith("liger_ce_fix")]:
        monkeypatch.delitem(sys.modules, name)
    mod = importlib.import_module("liger_ce_fix")
    assert hasattr(mod, "cross_entropy")
