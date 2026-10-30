import math
import pytest
import torch

from liger_ce_fix.reference import reference_cross_entropy
from tests.helpers import margin_logits

pytestmark = pytest.mark.cuda
V, H = 64_000, 256


def _fused():
    from liger_ce_fix.ops.fused_linear_cross_entropy import FusedLinearCrossEntropyFunction
    return FusedLinearCrossEntropyFunction


def _run_fused(h, w, y, reduction="mean", ignore_index=-100):
    h = h.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    loss = _fused().apply(h, w, y, ignore_index, reduction)
    loss.backward()
    return loss.detach().float(), h.grad.float(), w.grad.float()


def _run_reference(h, w, y, reduction="mean", ignore_index=-100):
    h = h.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    loss = reference_cross_entropy(h @ w.t(), y, ignore_index=ignore_index, reduction=reduction)
    loss.backward()
    return loss.detach().float(), h.grad.float(), w.grad.float()


@pytest.mark.parametrize("rows", [4096, 4097])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_fused_matches_reference_fp32(rows, reduction):
    torch.manual_seed(0)
    h = torch.randn(rows, H, device="cuda")
    w = torch.randn(V, H, device="cuda") * 0.05
    y = torch.randint(0, V, (rows,), device="cuda")
    y[::13] = -100
    loss, gh, gw = _run_fused(h, w, y, reduction)
    want, rgh, rgw = _run_reference(h, w, y, reduction)
    assert torch.allclose(loss, want, rtol=0, atol=1e-6 * max(1.0, want.abs().item()))
    assert torch.allclose(gh, rgh, rtol=1e-5, atol=1e-6)
    assert torch.allclose(gw, rgw, rtol=1e-5, atol=1e-6)   # grad_W: 128 chunked fp32 partial matmuls vs one


def test_fused_bf16_matches_reference_on_the_same_bf16_logits():
    torch.manual_seed(1)
    h = (torch.randn(2048, H, device="cuda") * 1.0).bfloat16()
    w = (torch.randn(V, H, device="cuda") * 0.05).bfloat16()
    y = torch.randint(0, V, (2048,), device="cuda")
    loss, gh, gw = _run_fused(h, w, y, "mean")
    # reference on the identical bf16 matmul, upcast afterwards
    h2 = h.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    want = reference_cross_entropy((h2 @ w2.t()).float(), y, reduction="mean")
    want.backward()
    assert torch.allclose(loss, want.detach().float(), rtol=1e-3)   # cuBLAS may round a few bf16 logits differently per chunk shape
    assert torch.allclose(gh, h2.grad.float(), rtol=2e-2, atol=1e-6)
    assert torch.allclose(gw, w2.grad.float(), rtol=2e-2, atol=1e-6)


@pytest.mark.parametrize("m", [20.0, 40.0, 80.0])
def test_fused_extreme_margin_through_identity_head(m):
    # weight = I[V_small] so logits == hidden; V_small keeps the identity head cheap
    Vs = 1024
    x, y = margin_logits(Vs, m, target=7, device="cuda")
    w = torch.eye(Vs, device="cuda")
    loss, gh, gw = _run_fused(x, w, y, "sum")
    expected = math.exp(-m)
    assert abs(loss.item() - expected) / expected <= 1e-3
    assert abs(gh[0, 7].item() + expected) / expected <= 1e-3
    assert gh.sum().abs().item() <= 1e-3 * expected


def test_fused_all_ignored_is_connected_zero():
    h = torch.randn(4, H, device="cuda")
    w = torch.randn(V, H, device="cuda")
    y = torch.full((4,), -100, device="cuda")
    loss, gh, gw = _run_fused(h, w, y, "mean")
    assert loss.item() == 0.0 and torch.equal(gh, torch.zeros_like(gh)) and torch.equal(gw, torch.zeros_like(gw))


def test_fused_backward_scales_with_grad_output():
    torch.manual_seed(2)
    h = torch.randn(64, H, device="cuda")
    w = torch.randn(3000, H, device="cuda") * 0.1
    y = torch.randint(0, 3000, (64,), device="cuda")
    h1, w1 = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    (_fused().apply(h1, w1, y, -100, "sum") * 0.3).backward()
    h2, w2 = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    _fused().apply(h2, w2, y, -100, "sum").backward()
    assert torch.allclose(h1.grad, 0.3 * h2.grad, rtol=1e-6, atol=1e-9)
    assert torch.allclose(w1.grad, 0.3 * w2.grad, rtol=1e-6, atol=1e-9)


def test_fused_rejects_reduction_none():
    h = torch.randn(4, H, device="cuda", requires_grad=True)
    w = torch.randn(100, H, device="cuda", requires_grad=True)
    with pytest.raises(ValueError):
        _fused().apply(h, w, torch.zeros(4, dtype=torch.int64, device="cuda"), -100, "none")


def test_fused_without_grad_requirement_returns_loss_only():
    h = torch.randn(8, H, device="cuda")
    w = torch.randn(500, H, device="cuda")
    y = torch.randint(0, 500, (8,), device="cuda")
    with torch.no_grad():
        loss = _fused().apply(h, w, y, -100, "mean")
    want = reference_cross_entropy(h @ w.t(), y)
    assert torch.allclose(loss, want, rtol=1e-6, atol=1e-6)


def test_fused_under_no_grad_with_parameter_weight_allocates_no_gradient_buffers():
    from liger_ce_fix import fused_linear_cross_entropy
    torch.manual_seed(3)
    h = torch.randn(2048, H, device="cuda")
    w = torch.nn.Parameter(torch.randn(V, H, device="cuda") * 0.05)   # the eval-time shape: frozen trunk, Parameter head
    y = torch.randint(0, V, (2048,), device="cuda")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.no_grad():
        loss = fused_linear_cross_entropy(h, w, y, reduction="mean")
    peak_delta = torch.cuda.max_memory_allocated() - before
    # a grad_weight buffer alone would be V*H*4 = 65.5 MB (+ a 65.5 MB fp32 temp); the logits chunk is 16 x V x 4 = 4 MB
    assert peak_delta < 32 * 1024 * 1024, peak_delta
    assert not loss.requires_grad
    assert torch.allclose(loss, reference_cross_entropy(h @ w.detach().t(), y), rtol=1e-6, atol=1e-6)
