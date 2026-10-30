"""The autograd-vs-plain-forward decision must be made where grad mode is visible (outside Function.forward).
A fake kernel records HAS_GRADIENTS so these run on CPU; the chunking / matmul around it is plain torch."""
import pytest
import torch

pytest.importorskip("triton", reason="the ops modules import triton at module level; CI's CPU wheel has none")
import liger_ce_fix.ops.cross_entropy as ce_mod  # noqa: E402
import liger_ce_fix.ops.fused_linear_cross_entropy as fused_mod
from liger_ce_fix.ops.cross_entropy import cross_entropy_autograd
from liger_ce_fix.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_autograd


class _Recorder:
    def __init__(self):
        self.has_gradients = []

    def __getitem__(self, grid):
        def launch(**kw):
            self.has_gradients.append(kw["HAS_GRADIENTS"])
        return launch


@pytest.fixture
def fake_kernel(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(ce_mod, "ce_fix_kernel", rec)
    monkeypatch.setattr(fused_mod, "ce_fix_kernel", rec)
    return rec


def _inputs():
    torch.manual_seed(0)
    h = torch.randn(8, 16)
    w = torch.nn.Parameter(torch.randn(50, 16))
    y = torch.randint(0, 50, (8,))
    return h, w, y


def test_fused_under_no_grad_skips_the_gradient_pass_even_for_a_parameter(fake_kernel):
    h, w, y = _inputs()
    with torch.no_grad():
        loss = fused_linear_cross_entropy_autograd(h, w, y, -100, "mean")
    assert fake_kernel.has_gradients and not any(fake_kernel.has_gradients)
    assert not loss.requires_grad


def test_fused_with_grad_enabled_runs_the_gradient_pass(fake_kernel):
    h, w, y = _inputs()
    loss = fused_linear_cross_entropy_autograd(h, w, y, -100, "mean")
    assert fake_kernel.has_gradients and all(fake_kernel.has_gradients)
    assert loss.requires_grad


def test_fused_without_any_requires_grad_skips_the_gradient_pass(fake_kernel):
    h, w, y = _inputs()
    loss = fused_linear_cross_entropy_autograd(h, w.detach(), y, -100, "sum")
    assert fake_kernel.has_gradients and not any(fake_kernel.has_gradients)
    assert not loss.requires_grad


def test_cross_entropy_under_no_grad_does_not_request_gradients(fake_kernel):
    x = torch.randn(4, 10, requires_grad=True)
    y = torch.randint(0, 10, (4,))
    with torch.no_grad():
        loss = cross_entropy_autograd(x, y, -100, "sum")
    assert fake_kernel.has_gradients == [False] and not loss.requires_grad


def test_cross_entropy_with_grad_enabled_requests_gradients(fake_kernel):
    x = torch.randn(4, 10, requires_grad=True)
    y = torch.randint(0, 10, (4,))
    loss = cross_entropy_autograd(x, y, -100, "sum")
    assert fake_kernel.has_gradients == [True] and loss.requires_grad
