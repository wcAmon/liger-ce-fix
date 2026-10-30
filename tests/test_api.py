import pytest
import torch
import torch.nn.functional as F

import liger_ce_fix
from liger_ce_fix import cross_entropy, fused_linear_cross_entropy, reference_cross_entropy

UNSUPPORTED = dict(bias=torch.zeros(5), ce_weight=torch.ones(5), softcap=30.0, lse_square_scale=1e-4,
                   label_smoothing=0.1, return_z_loss=True, return_token_accuracy=True,
                   return_predicted_tokens=True, accum_dtype=torch.float32, use_token_scaling=True)


def test_exports():
    assert liger_ce_fix.__all__ == ["fused_linear_cross_entropy", "cross_entropy", "reference_cross_entropy",
                                    "__version__", "__commit__"]


def test_cross_entropy_on_cpu_is_the_reference():
    torch.manual_seed(0)
    x = torch.randn(7, 50)
    y = torch.randint(0, 50, (7,))
    assert torch.equal(cross_entropy(x, y, reduction="sum"), reference_cross_entropy(x, y, reduction="sum"))
    assert torch.allclose(cross_entropy(x, y), F.cross_entropy(x, y), rtol=1e-6)


def test_fused_on_cpu_raises_with_pointer_to_reference():
    with pytest.raises(RuntimeError, match="reference_cross_entropy"):
        fused_linear_cross_entropy(torch.randn(3, 8), torch.randn(20, 8), torch.zeros(3, dtype=torch.int64))


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unsupported_options_raise_value_error(name):
    x = torch.randn(2, 5)
    y = torch.tensor([0, 1])
    with pytest.raises(ValueError, match=name):
        cross_entropy(x, y, **{name: UNSUPPORTED[name]})
    with pytest.raises(ValueError, match=name):
        fused_linear_cross_entropy(torch.randn(2, 4), torch.randn(5, 4), y, **{name: UNSUPPORTED[name]})


def test_unknown_keyword_also_raises():
    with pytest.raises(ValueError, match="frobnicate"):
        cross_entropy(torch.randn(2, 5), torch.tensor([0, 1]), frobnicate=1)


def test_bad_reduction_raises_before_dispatch():
    with pytest.raises(ValueError):
        cross_entropy(torch.randn(2, 5), torch.tensor([0, 1]), reduction="avg")


@pytest.mark.cuda
def test_cross_entropy_on_cuda_uses_kernel_and_matches_reference():
    torch.manual_seed(1)
    x = torch.randn(33, 777, device="cuda", requires_grad=True)
    y = torch.randint(0, 777, (33,), device="cuda")
    loss = cross_entropy(x, y, reduction="none")
    assert loss.shape == (33,)
    want = reference_cross_entropy(x.detach(), y, reduction="none")
    assert torch.allclose(loss, want, rtol=1e-6, atol=1e-6)


@pytest.mark.cuda
def test_fused_on_cuda_matches_reference():
    torch.manual_seed(2)
    h = torch.randn(16, 32, device="cuda", requires_grad=True)
    w = torch.randn(600, 32, device="cuda", requires_grad=True)
    y = torch.randint(0, 600, (16,), device="cuda")
    loss = fused_linear_cross_entropy(h, w, y, reduction="sum")
    want = reference_cross_entropy(h.detach() @ w.detach().t(), y, reduction="sum")
    assert torch.allclose(loss, want, rtol=1e-6, atol=1e-6)
