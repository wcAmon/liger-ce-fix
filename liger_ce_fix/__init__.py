"""Liger fused linear cross-entropy with the log-space softmax-collapse fix (arXiv 2610.31642).

Three entry points, one formula (see reference.py):
  fused_linear_cross_entropy(hidden, weight, target)  CUDA only; logits never materialized
  cross_entropy(logits, target)                        CUDA -> Triton kernel, CPU -> reference
  reference_cross_entropy(logits, target)              pure torch, any device
Options Liger offers but this fork dropped (bias, class weights, softcapping, z-loss, label smoothing, accuracy,
token scaling, accum_dtype) raise ValueError instead of being ignored."""
import torch

from liger_ce_fix._meta import commit as _commit, version as _version
from liger_ce_fix.reference import check_reduction, reference_cross_entropy

__version__ = _version()
__commit__ = _commit()

_DROPPED = ("bias", "ce_weight", "softcap", "lse_square_scale", "label_smoothing", "return_z_loss",
            "return_token_accuracy", "return_predicted_tokens", "accum_dtype", "use_token_scaling")


def _reject(extra: dict) -> None:
    if extra:
        name = next(iter(extra))
        hint = "not supported by liger-ce-fix" if name in _DROPPED else "unknown option"
        raise ValueError(f"{name}: {hint} (supported keywords: ignore_index, reduction)")


def cross_entropy(logits: torch.Tensor, target: torch.Tensor, *, ignore_index: int = -100,
                  reduction: str = "mean", **extra) -> torch.Tensor:
    """Cross-entropy over materialized logits [N, V]. On CUDA the Triton kernel overwrites `logits` with the gradient
    (do not reuse the tensor); on CPU the pure-torch reference runs."""
    _reject(extra)
    check_reduction(reduction)
    if logits.is_cuda:
        from liger_ce_fix.ops.cross_entropy import CrossEntropyFunction
        return CrossEntropyFunction.apply(logits, target, ignore_index, reduction)
    return reference_cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)


def fused_linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, *,
                               ignore_index: int = -100, reduction: str = "mean", **extra) -> torch.Tensor:
    """Fused head matmul + cross-entropy: hidden [N, H], weight [V, H], target [N]. CUDA only; reduction 'mean' or
    'sum'. For CPU use reference_cross_entropy(hidden @ weight.T, target)."""
    _reject(extra)
    check_reduction(reduction)
    if not hidden.is_cuda:
        raise RuntimeError("fused_linear_cross_entropy needs CUDA tensors; on CPU call "
                           "reference_cross_entropy(hidden @ weight.T, target, ...)")
    from liger_ce_fix.ops.fused_linear_cross_entropy import FusedLinearCrossEntropyFunction
    return FusedLinearCrossEntropyFunction.apply(hidden, weight, target, ignore_index, reduction)


__all__ = ["fused_linear_cross_entropy", "cross_entropy", "reference_cross_entropy", "__version__", "__commit__"]
