import math
import pytest
import torch
import torch.nn.functional as F

from liger_ce_fix.reference import reference_cross_entropy
from tests.helpers import margin_logits

pytestmark = pytest.mark.cuda
V = 64_000


def _kernel():
    from liger_ce_fix.ops.cross_entropy import CrossEntropyFunction
    return CrossEntropyFunction


def _grad_kernel(x, y, reduction="mean", ignore_index=-100, dtype=torch.float32):
    x = x.cuda().to(dtype).detach().clone().requires_grad_(True)
    out = _kernel().apply(x, y.cuda(), ignore_index, reduction)
    out.sum().backward()
    return out.detach().float(), x.grad.float()


def _grad_reference(x, y, reduction="mean", ignore_index=-100, dtype=torch.float32):
    x = x.cuda().to(dtype).detach().clone().requires_grad_(True)
    out = reference_cross_entropy(x, y.cuda(), ignore_index=ignore_index, reduction=reduction)
    out.sum().backward()
    return out.detach().float(), x.grad.float()


@pytest.mark.parametrize("rows", [4096, 4097])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_kernel_matches_reference_on_random_logits(rows, reduction):
    torch.manual_seed(0)
    x = torch.randn(rows, V) * 3
    y = torch.randint(0, V, (rows,))
    y[::17] = -100
    got, g_got = _grad_kernel(x, y, reduction)
    want, g_want = _grad_reference(x, y, reduction)
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-6)
    assert torch.allclose(g_got, g_want, rtol=1e-5, atol=1e-9)


def test_kernel_non_power_of_two_vocab():
    torch.manual_seed(3)
    x = torch.randn(33, 1000)
    y = torch.randint(0, 1000, (33,))
    got, g_got = _grad_kernel(x, y, "sum")
    want, g_want = _grad_reference(x, y, "sum")
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-6) and torch.allclose(g_got, g_want, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("m", [20.0, 40.0, 80.0])
def test_kernel_extreme_margin(m):
    x, y = margin_logits(V, m, target=7)
    loss, g = _grad_kernel(x, y, "sum")
    expected = math.exp(-m)
    assert abs(loss.item() - expected) / expected <= 1e-3
    assert abs(g[0, 7].item() + expected) / expected <= 1e-3
    assert g.sum().abs().item() <= 1e-9 and g.sum().abs().item() <= 1e-3 * expected


def test_kernel_margin_200_is_finite():
    x, y = margin_logits(V, 200.0, target=3)
    loss, g = _grad_kernel(x, y, "sum")
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    assert abs(g[0, 3].item()) <= 1.2e-38 and g.sum().abs().item() <= 1e-9


def test_kernel_negative_margin():
    x, y = margin_logits(V, -50.0, target=11)
    loss, g = _grad_kernel(x, y, "sum")
    assert abs(loss.item() - 50.0) / 50.0 <= 1e-6 and abs(g[0, 11].item() + 1.0) <= 1e-6


def test_kernel_bf16_logits_keep_tiny_target_gradient():
    x, y = margin_logits(V, 20.0, target=5)
    loss, g = _grad_kernel(x, y, "sum", dtype=torch.bfloat16)
    want, g_want = _grad_reference(x, y, "sum", dtype=torch.bfloat16)
    assert g[0, 5].item() != 0.0 and g[0, 5].item() < 0.0
    assert torch.allclose(loss, want, rtol=1e-2) and torch.allclose(g, g_want, rtol=1e-2, atol=1e-12)


def test_kernel_custom_ignore_index():
    torch.manual_seed(4)
    x = torch.randn(8, 300)
    y = torch.tensor([0, 1, 2, 0, 3, 299, 0, 4])
    for ignore in (0, 299):
        got, g_got = _grad_kernel(x, y, "mean", ignore_index=ignore)
        want, g_want = _grad_reference(x, y, "mean", ignore_index=ignore)
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-6) and torch.allclose(g_got, g_want, rtol=1e-5, atol=1e-9)
        assert torch.equal(g_got[y == ignore], torch.zeros_like(g_got[y == ignore]))


def test_kernel_all_ignored_returns_zero_and_zero_grad():
    x = torch.randn(5, 64)
    y = torch.full((5,), -100)
    got, g = _grad_kernel(x, y, "mean")
    assert got.item() == 0.0 and torch.equal(g, torch.zeros_like(g))


def test_kernel_accepts_noncontiguous_logits_and_target():
    torch.manual_seed(5)
    base = torch.randn(V, 40)            # transposed storage
    x = base.t()                          # [40, V], non-contiguous
    y_all = torch.randint(0, V, (80,))
    y = y_all[::2]                        # strided target
    got, g_got = _grad_kernel(x.contiguous(), y, "sum")
    x_nc = x.cuda().detach().clone().t().contiguous().t().requires_grad_(True)   # keep it non-contiguous on device
    out = _kernel().apply(x_nc, y.cuda()[::1], -100, "sum")
    out.backward()
    assert torch.allclose(out.float(), got, rtol=1e-6, atol=1e-6)
    assert torch.allclose(x_nc.grad.float(), g_got, rtol=1e-5, atol=1e-9)


def test_kernel_backward_scales_with_grad_output():
    torch.manual_seed(6)
    x = torch.randn(16, 500).cuda()
    y = torch.randint(0, 500, (16,)).cuda()
    x1 = x.clone().requires_grad_(True)
    (_kernel().apply(x1, y, -100, "sum") * 0.25).backward()
    x2 = x.clone().requires_grad_(True)
    _kernel().apply(x2, y, -100, "sum").backward()
    assert torch.allclose(x1.grad, 0.25 * x2.grad, rtol=1e-6, atol=1e-9)
    # per-row grad_output for reduction="none"
    x3 = x.clone().requires_grad_(True)
    w = torch.linspace(0.1, 1.6, 16).cuda()
    (_kernel().apply(x3, y, -100, "none") * w).sum().backward()
    assert torch.allclose(x3.grad, x2.grad * w[:, None], rtol=1e-6, atol=1e-9)


def test_kernel_all_other_logits_minus_inf_gives_zero_loss_and_finite_grad():
    x = torch.full((2, 5), float("-inf"))
    x[0, 2] = 1.0
    x[1, 4] = -3.0
    loss, g = _grad_kernel(x, torch.tensor([2, 4]), "sum")
    assert loss.item() == 0.0 and torch.isfinite(g).all()
    assert torch.equal(g, torch.zeros_like(g))
