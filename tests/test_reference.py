import math
import pytest
import torch
import torch.nn.functional as F

from liger_ce_fix.reference import reference_cross_entropy
from tests.helpers import margin_logits

V = 64_000


def _grad(fn, x, *args, **kw):
    x = x.detach().clone().requires_grad_(True)
    out = fn(x, *args, **kw)
    out.sum().backward()
    return out.detach(), x.grad


def test_matches_torch_cross_entropy_at_ordinary_margins():
    torch.manual_seed(0)
    x = torch.randn(64, V) * 3
    y = torch.randint(0, V, (64,))
    for reduction in ("mean", "sum", "none"):
        got, g_got = _grad(reference_cross_entropy, x, y, reduction=reduction)
        want, g_want = _grad(F.cross_entropy, x, y, reduction=reduction)
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-7), reduction
        assert torch.allclose(g_got, g_want, rtol=1e-5, atol=1e-9), reduction


@pytest.mark.parametrize("m", [20.0, 40.0, 80.0])
def test_extreme_positive_margin_keeps_target_gradient_where_torch_gives_zero(m):
    x, y = margin_logits(V, m, target=7)
    loss, g = _grad(reference_cross_entropy, x, y, reduction="sum")
    expected = math.exp(-m)
    assert loss.item() > 0 and abs(loss.item() - expected) / expected <= 1e-3
    assert abs(g[0, 7].item() + expected) / expected <= 1e-3
    assert g.sum().abs().item() <= 1e-9 and g.sum().abs().item() <= 1e-3 * expected
    # the contrast the paper describes: plain fp32 CE has already collapsed
    _, g_torch = _grad(F.cross_entropy, x, y, reduction="sum")
    assert g_torch[0, 7].item() == 0.0


def test_margin_200_underflows_gracefully():
    x, y = margin_logits(V, 200.0, target=3)
    loss, g = _grad(reference_cross_entropy, x, y, reduction="sum")
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    assert abs(g[0, 3].item()) <= 1.2e-38 and g.sum().abs().item() <= 1e-9


def test_negative_margin_is_the_plain_loss():
    x, y = margin_logits(V, -50.0, target=11)
    loss, g = _grad(reference_cross_entropy, x, y, reduction="sum")
    assert abs(loss.item() - 50.0) / 50.0 <= 1e-6
    assert abs(g[0, 11].item() + 1.0) <= 1e-6
    assert abs(g[0].sum().item()) <= 1e-6 and torch.isfinite(g).all()


def test_ignore_index_rows_contribute_nothing_and_mean_divides_by_kept_rows():
    torch.manual_seed(1)
    x = torch.randn(6, 100)
    y = torch.tensor([1, -100, 2, -100, 3, 4])
    loss, g = _grad(reference_cross_entropy, x, y, reduction="mean")
    want = F.cross_entropy(x[[0, 2, 4, 5]], y[[0, 2, 4, 5]])
    assert torch.allclose(loss, want, rtol=1e-6)
    assert torch.equal(g[1], torch.zeros(100)) and torch.equal(g[3], torch.zeros(100))
    per = reference_cross_entropy(x, y, reduction="none")
    assert per[1].item() == 0.0 and per[3].item() == 0.0 and per.shape == (6,)


def test_all_ignored_batch_returns_connected_zero():
    x = torch.randn(3, 10, requires_grad=True)
    y = torch.full((3,), -100)
    loss = reference_cross_entropy(x, y, reduction="mean")
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(x.grad, torch.zeros(3, 10))


def test_reference_custom_ignore_index():
    x = torch.randn(4, 10)
    y = torch.tensor([0, 9, 0, 2])
    loss0 = reference_cross_entropy(x, y, ignore_index=0, reduction="sum")
    want = F.cross_entropy(x[[1, 3]], y[[1, 3]], reduction="sum")
    assert torch.allclose(loss0, want, rtol=1e-6)
    loss9 = reference_cross_entropy(x, y, ignore_index=9, reduction="sum")
    assert torch.allclose(loss9, F.cross_entropy(x[[0, 2, 3]], y[[0, 2, 3]], reduction="sum"), rtol=1e-6)


def test_reductions_relate():
    torch.manual_seed(2)
    x = torch.randn(9, 50)
    y = torch.randint(0, 50, (9,))
    y[4] = -100
    none = reference_cross_entropy(x, y, reduction="none")
    assert torch.allclose(reference_cross_entropy(x, y, reduction="sum"), none.sum(), rtol=1e-6)
    assert torch.allclose(reference_cross_entropy(x, y, reduction="mean"), none.sum() / 8, rtol=1e-6)


def test_reference_all_other_logits_minus_inf_gives_zero_loss_and_finite_grad():
    x = torch.full((1, 5), float("-inf"))
    x[0, 2] = 1.0
    loss, g = _grad(reference_cross_entropy, x, torch.tensor([2]), reduction="sum")
    assert loss.item() == 0.0 and torch.isfinite(g).all()


def test_bad_arguments_raise():
    x = torch.randn(2, 5)
    with pytest.raises(ValueError):
        reference_cross_entropy(x, torch.tensor([0, 1]), reduction="avg")
    with pytest.raises(ValueError):
        reference_cross_entropy(x, torch.tensor([0, 5]))          # out of range
    with pytest.raises(ValueError):
        reference_cross_entropy(x, torch.tensor([0, 1, 2]))       # shape mismatch
    with pytest.raises(ValueError):
        reference_cross_entropy(x.unsqueeze(0), torch.tensor([0, 1]))  # not 2-D
